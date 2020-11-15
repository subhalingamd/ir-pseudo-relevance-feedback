import argparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import krovetz
import string
import re
from math import log
import json
ks = krovetz.PyKrovetzStemmer()
stop_words = set(stopwords.words('english')) # | set(string.punctuation)

#word_df = 1

def preprocess(text):
	word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9]',' ',text.lower()))
	processed = [ks.stem(w) for w in word_tokens]
	return processed



collection_file = "scratch/assg2/data/msmarco-docs.tsv"
docid_file_offset = {}
vocab_words_df = {}
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
		#doc_body_processed = set(doc_body_processed)
		docid_file_offset.update({line_comp[0]:offset})
		offset = f.tell()
		if len(doc_body_processed) > 0:
			vocab_words_df[doc_body_processed[0]] = vocab_words_df.get(doc_body_processed[0],0) + 1
			for i in range(1,len(doc_body_processed)):
				pair = doc_body_processed[i-1]+' '+doc_body_processed[i]
				vocab_words_df_pairs[pair] = vocab_words_df_pairs.get(pair,0) + 1
				vocab_words_df[doc_body_processed[i]] = vocab_words_df.get(doc_body_processed[i],0) + 1
			

#avg_doc_len_coll = tot_doc_len/len(docid_file_offset) if len(docid_file_offset)>0 else 0 # avoid divide by zero

print(len(docid_file_offset),len(vocab_words_df),tot_doc_len)


with open('scratch/bi-df.json', 'w') as fp:
	json.dump(vocab_words_df,fp)
with open('scratch/bi-df_pairs.json', 'w') as fp:
	json.dump(vocab_words_df_pairs,fp)
with open('scratch/bi-offset.json', 'w') as fp:
	json.dump(docid_file_offset,fp)
