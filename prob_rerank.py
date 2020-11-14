import argparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import krovetz
import string
import re
from math import log
ks = krovetz.PyKrovetzStemmer()
stop_words = set(stopwords.words('english')) # | set(string.punctuation)

#word_df = 1

def preprocess(text):
	word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9]',' ',text.lower()))
	processed = [ks.stem(w) for w in word_tokens if w not in stop_words]
	return processed

def bm25(qtext,docs_id,docs_body,vocab_words_df,num_docs_collection,avg_docs_len):
	rel_scores = [0 for i in range(len(docs_id))]
	N = num_docs_collection
	k1 = 1.4
	b = 0.75
	for q in qtext:
		n_q = vocab_words_df.get(q,0)
		idf_q = log(( (N-n_q+0.5)/(n_q+0.5) + 1))
		idf_q = idf_q if idf_q > 0 else 1
		for i in range(len(docs_id)):
			# docid = docs_id[i]
			docbody = docs_body[i]
			f_q_d = docbody.count(q)
			len_d = len(docbody)
			curr_rel = idf_q * f_q_d * (k1+1) / (f_q_d + k1*(1-b+(b*len_d/avg_docs_len)))
			rel_scores[i] += curr_rel

	reranked_docs = [(doc,_) for _, doc in sorted(zip(rel_scores,docs_id), key=lambda x: x[0], reverse=True)]
	return reranked_docs

def output_work(qid,reranked_docs,filepath):
	with open(filepath,'a') as f:
		curr_rank = 1
		for item in reranked_docs:
			print(qid,'Q0',item[0],curr_rank,item[1],'runid1',file=f)
			curr_rank += 1
		print("\n",file=f)

def do_task(docid_file_offset,qtext,result_docs,collection_file,expansion_limit,vocab_words_df,avg_doc_len_coll):
	percent_rel = 0.35
	rel_docs_ct = int(percent_rel * len(result_docs))
	qtext = preprocess(qtext)
	qtext_set = set(qtext)
	df_rel_doc_set, df_all_doc_set = {},{}
	doc_body_all = []
	i = 0
	tot_doc_len = 0
	with open(collection_file,'r',encoding="utf-8") as f:
		for docid in result_docs:
			"""
			#test
			if docid not in docid_file_offset:
				doc_body_all.append([])
				continue
			#end test
			"""
			docid_seek = docid_file_offset[docid]
			f.seek(docid_seek)
			doc_data = f.readline()
			doc_body = doc_data.rstrip('\n').split('\t')[-1]
			doc_body = preprocess(doc_body)
			tot_doc_len += len(doc_body)
			doc_body_all.append(doc_body)
			doc_body_set = set(doc_body)
			for word in doc_body_set:
				if word in qtext_set:
					continue
				df_all_doc_set[word] = df_all_doc_set.get(word,0) + 1
				if i < rel_docs_ct:
					df_rel_doc_set[word] = df_rel_doc_set.get(word,0) + 1
				i += 1

	vocab_words_df = df_all_doc_set
	# Define variables as in paper
	N = len(result_docs) 	# the number of documents in the collection
	R = rel_docs_ct 		# the number of known relevant document for a request = 100
	# r = the number of known relevant documents term t(i) occurs in
	# n = the number of documents term t(i) occurs in = df(i)

	for word,n in vocab_words_df.items():
		r = df_rel_doc_set.get(word,0)
		score = r * log ( ( (r+0.5)*(N-n-R+r+0.5) ) / ( (n-r+0.5)*(R-r+0.5) ) )
		# just update the values
		vocab_words_df[word] = score

	# add query terms
	new_queries = sorted(vocab_words_df.items(),key=lambda x:x[1], reverse = True)[:expansion_limit]
	qtext.extend([qw_[0] for qw_ in new_queries])

	return bm25(qtext=qtext,docs_id=result_docs,docs_body=doc_body_all,vocab_words_df=vocab_words_df,num_docs_collection=N,avg_docs_len=tot_doc_len/N+1e-4)


def prob_rerank_method(collection_file,top_100_file,expansion_limit,query_file,output_file):
	
	# open a new file if output already exists
	with open(output_file,'w') as f:
		pass

	docid_file_offset = {}
	#vocab_words_df = {}
	offset = 0
	#tot_doc_len = 0
	with open(collection_file,'r',encoding="utf-8") as f:
		#for line in f:
		while 1:
			line = f.readline()
			if not line:
				break
			line_comp = line.rstrip('\n').split('\t')
			#doc_body_processed = preprocess(line_comp[-1])
			#tot_doc_len += len(doc_body_processed)
			#doc_body_processed = set(doc_body_processed)
			#for word in doc_body_processed:
			#	vocab_words_df[word] = vocab_words_df.get(word,0) + 1
			docid_file_offset.update({line_comp[0]:offset})
			offset = f.tell()

	#avg_doc_len_coll = tot_doc_len/len(docid_file_offset) if len(docid_file_offset)>0 else 0 # avoid divide by zero

	with open(query_file,'r',encoding="utf-8") as f:
		qline = f.readline()
		qline_comp = qline.rstrip('\n').split('\t')
		result_docs = []
		query_count = 0
		with open(top_100_file,'r',encoding="utf-8") as f100:
			while 1:
			#for line100 in f100:
				line100 = f100.readline()
				line100_comp = line100.rstrip('\n').split()

				if not line100 or line100_comp[0] != qline_comp[0]:
					#assert query_count == 100
					qtext = qline_comp[1]

					# process
					reranked_docs = do_task(docid_file_offset=docid_file_offset,qtext=qtext,result_docs=result_docs,collection_file=collection_file,expansion_limit=expansion_limit,vocab_words_df=None,avg_doc_len_coll=1)
					output_work(qid=qline_comp[0],reranked_docs=reranked_docs,filepath=output_file)

					query_count = 0
					result_docs = []
					qline = f.readline()
					if not qline:
						break
					qline_comp = qline.rstrip('\n').split('\t')

				result_docs.append(line100_comp[2])
				query_count += 1
					


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Probabilistic Retrieval Reranking')
	parser.add_argument('query_file', metavar='query-file',
                    help='file containing the queries in the same tsv format as given in Table 1 for queries file')
	parser.add_argument('top_100_file', metavar='top-100-file',
                    help='a file containing the top100 documents in the same format as train and dev top100 files given, which need to be reranked')
	parser.add_argument('collection_file', metavar='collection-file',
                    help='file containing the full document collection (in the same format as msmarco-docs file given)')
	parser.add_argument('expansion_limit', metavar='expansion-limit', type=int, choices=range(1,15+1),
                    help='is a number ranging from 1-15 that specifies the limit on the number of additional terms in the expanded query')
	# not part of specs
	parser.add_argument('-o','--output', metavar="resultfile", default="out_reranked", 
                    help='the output file named resultfile which is generated by your program after reranking')
	args = parser.parse_args()

	prob_rerank_method(collection_file=args.collection_file,top_100_file=args.top_100_file,expansion_limit=args.expansion_limit,query_file=args.query_file,output_file=args.output)


