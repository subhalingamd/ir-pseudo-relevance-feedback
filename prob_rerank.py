import argparse

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
	args = parser.parse_args()

	query_file = args.query_file