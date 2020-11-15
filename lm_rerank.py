import argparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import krovetz
import string
import re
from math import log
ks = krovetz.PyKrovetzStemmer()
stop_words = set(stopwords.words('english')) # | set(string.punctuation)

def preprocess(text):
	word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9]',' ',text.lower()))
	processed = [ks.stem(w) for w in word_tokens if w not in stop_words]
	return processed

def preprocess_with_stop(text):
	word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9]',' ',text.lower()))
	processed = [ks.stem(w) for w in word_tokens]
	return processed

def output_work(qid,reranked_docs,filepath):
	with open(filepath,'a') as f:
		curr_rank = 1
		for item in reranked_docs:
			print(qid,'Q0',item[0],curr_rank,item[1],'runid1',file=f)
			curr_rank += 1
		print("\n",file=f)

def uni_dirch_smooth(t,docbody,vocab_words_df,len_c):
	# P(t|D) = (f{t,d} + \mu*P_c(t))/(|D| + \mu)
	# P_c(t) = f{t,C}/|C|
	mu = 2
	f_td = docbody.count(t)
	len_d = len(docbody)
	f_tc = vocab_words_df.get(t,0)
	len_c = len_c

	p_c = f_tc/len_c
	p_final = (f_td + mu*p_c)/(len_d + mu)
	return p_final


def uni_lavrenko_croft(docs_body,vocab_words_df,len_c,q_text):
	# P(w|R) constant for given word irrespective of doc
	# docs_body like [["A1","A2"],["B1"]]
	# qtext like ["word1","word2","word3"]

	dict_p_w_R = {}
	for w in vocab_words_df:  # find P(w|R) for given word w in vocabulary
		p_wQ = 0

		for docbody in docs_body:
			p_m = 1 # Assume uniform distribution
			
			p_w_M = uni_dirch_smooth(w,docbody,vocab_words_df,len_c)
			prod_p_q_M = 1
			for q in q_text:
				prod_p_q_M *= uni_dirch_smooth(q,docbody,vocab_words_df,len_c)

			#p_wQ_M = p_w_M * prod_p_q_M

			p_wQ += (p_m * p_w_M * prod_p_q_M)

		dict_p_w_R.update({w:p_wQ})
	
	return dict_p_w_R


def do_uni_task(docid_file_offset,qtext,result_docs,collection_file,vocab_words_df,tot_doc_len):
	# rank using KL divergence score
	
	#this will be added using append
	# rel_scores = [0 for i in range(len(result_docs))]
	rel_scores = []
	docs_id = result_docs

	qtext = preprocess(qtext)
	doc_body_all = []
	with open(collection_file,'r',encoding="utf-8") as f:
		for docid in result_docs:
			docid_seek = docid_file_offset[docid]
			f.seek(docid_seek)
			doc_data = f.readline()
			doc_body = doc_data.rstrip('\n').split('\t')[-1]
			doc_body = preprocess(doc_body)
			doc_body_all.append(doc_body)

	dict_p_w_R = uni_lavrenko_croft(docs_body=doc_body_all,vocab_words_df=vocab_words_df,len_c=tot_doc_len,q_text=qtext)
	
	# use KL diveregnce
	for docbody in doc_body_all:
		curr_score = 0
		for w,p_w_R in dict_p_w_R.items():
			curr_score += (p_w_R * log(uni_dirch_smooth(w,docbody,vocab_words_df,len_c=tot_doc_len)))
		rel_scores.append(curr_score)

	reranked_docs = [(doc,_) for _, doc in sorted(zip(rel_scores,docs_id), key=lambda x: x[0], reverse=True)]
	return reranked_docs

def do_bi_task(docid_file_offset,qtext,result_docs,collection_file,vocab_words_df,vocab_words_df_pairs,tot_doc_len):
	return None


#### [mod]
#### verified below

def uni_lm_rerank_method(collection_file,top_100_file,query_file,output_file):
	
	# open a new file if output already exists
	with open(output_file,'w') as f:
		pass

	docid_file_offset = {}
	vocab_words_df = {}  # here it is frequency!
	offset = 0
	tot_doc_len = 0
	with open(collection_file,'r',encoding="utf-8") as f:
		#for line in f:
		while 1:
			line = f.readline()
			if not line:
				break
			line_comp = line.rstrip('\n').split('\t')
			doc_body_processed = preprocess(line_comp[-1])
			tot_doc_len += len(doc_body_processed)
			#[mod] doc_body_processed = set(doc_body_processed)
			for word in doc_body_processed:
				vocab_words_df[word] = vocab_words_df.get(word,0) + 1
			docid_file_offset.update({line_comp[0]:offset})
			offset = f.tell()

	#[mod] next line is not req!!!
	# avg_doc_len_coll = tot_doc_len/len(docid_file_offset) if len(docid_file_offset)>0 else 0 # avoid divide by zero
	
	"""
	import json
	with open("scratch/smooth-df.json", 'r') as fp:
		vocab_words_df = json.load(fp)
	with open("scratch/smooth-offset.json", 'r') as fp:
		docid_file_offset = json.load(fp)
	tot_doc_len = 2385306469
	"""

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
					reranked_docs = do_uni_task(docid_file_offset=docid_file_offset,qtext=qtext,result_docs=result_docs,collection_file=collection_file,vocab_words_df=vocab_words_df,tot_doc_len=tot_doc_len)
					output_work(qid=qline_comp[0],reranked_docs=reranked_docs,filepath=output_file)

					query_count = 0
					result_docs = []
					qline = f.readline()
					if not qline:
						break
					qline_comp = qline.rstrip('\n').split('\t')

				result_docs.append(line100_comp[2])
				query_count += 1
					



def bi_lm_rerank_method(collection_file,top_100_file,query_file,output_file):
	
	# open a new file if output already exists
	with open(output_file,'w') as f:
		pass

	docid_file_offset = {}
	vocab_words_df = {}  # here it is frequency!
	vocab_words_df_pairs = {}
	offset = 0
	tot_doc_len = 0
	with open(collection_file,'r',encoding="utf-8") as f:
		#for line in f:
		while 1:
			line = f.readline()
			if not line:
				break
			line_comp = line.rstrip('\n').split('\t')
			doc_body_processed = preprocess(line_comp[-1])
			tot_doc_len += len(doc_body_processed)
			#[mod] doc_body_processed = set(doc_body_processed)
			if len(doc_body_processed) > 0:
				vocab_words_df[doc_body_processed[0]] = vocab_words_df.get(doc_body_processed[0],0) + 1
				for i in range(1,len(doc_body_processed)):
					pair = doc_body_processed[i-1]+' '+doc_body_processed[i]
					vocab_words_df_pairs[pair] = vocab_words_df_pairs.get(pair,0) + 1
					vocab_words_df[doc_body_processed[i]] = vocab_words_df.get(doc_body_processed[i],0) + 1
			docid_file_offset.update({line_comp[0]:offset})
			offset = f.tell()

	#[mod] next line is not req!!!
	# avg_doc_len_coll = tot_doc_len/len(docid_file_offset) if len(docid_file_offset)>0 else 0 # avoid divide by zero

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
					reranked_docs = do_bi_task(docid_file_offset=docid_file_offset,qtext=qtext,result_docs=result_docs,collection_file=collection_file,vocab_words_df=vocab_words_df,vocab_words_df_pairs=vocab_words_df_pairs,tot_doc_len=tot_doc_len)
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
	parser.add_argument('model', metavar='model', choices=('uni','bi'),
                    help='it specifies the unigram or the bigram language model that should be used for relevance language model')
	# not part of specs
	parser.add_argument('-o','--output', metavar="resultfile", default="out_reranked", 
                    help='the output file named resultfile which is generated by your program after reranking')
	args = parser.parse_args()

	if args.model == 'bi':
		bi_lm_rerank_method(collection_file=args.collection_file,top_100_file=args.top_100_file,query_file=args.query_file,output_file=args.output)
	else:
		uni_lm_rerank_method(collection_file=args.collection_file,top_100_file=args.top_100_file,query_file=args.query_file,output_file=args.output)

