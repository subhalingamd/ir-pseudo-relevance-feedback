# document-reranking
Telescoping models aimed at improving the precision of results using pseudo-relevance feedback

## Introduction
There are *3213835* documents in the dataset *(named **MS-MARCO**)* which is based on a collection from *Microsoft Bing*. The collection is in a single *tsv* file in the format: `docid`, `url`, `title`, `body`. **The body has already been pre-processed to remove the HTML tags.** The queries are given in a seperate *tsv* file in the format: `query_id` and `query`.

The top-100 documents for each query retrieved from a BM25 model has already been provided (in the same format supported by *trec_eval*) and the aim of this assignment is to reorder/rerank these documents so that more relevant documents appear first using two methods: **Probabilistic Retrieval Query expansion** and **Relevance Model based Language Modeling**.

## Requirements
**Python** *(recommended 3.6+)* is the language used. The dataset used is a large one (around 22GB+), so make sure that you have sufficient storage!

The project uses `nltk` for tokenizing & stop-word removal and `krovetz` which is a python wrapper for *KrovetzStemmer*. The complete set of libraries (which includes dependencies of the above mentioned libraries) may be installed using `requirements.txt` with the following command:

```
pip3 install -r requirements.txt
```

`nltk` additionally requires `stopwords` corpora for stop-words removal. To download it, you can use the following code:
```python 
import nltk
nltk.download("stopwords")
```
*(You can also use `nltk.download()`, which gives a GUI to download modules)*

To ease the installation, you can make use of the `makefile` provided. You can use the following command:
```
make
```

***Note that there were some errors when `krovetz` was used with Python 3.9. Even though the exact cause for the error is not known, kindly switch to an older version if you face such issues.*** *It was tested with Python 3.7.2 without any problems.*



## File Structure
The main source files are 
- [`prob_rerank.py`](#prob_rerankpy): Probabilistic Retrieval Query expansion
- [`lm_rerank.py`](#lm_rerankpy): Relevance Model based Language Modeling

### `prob_rerank.py`
This program uses query expansion (ranging from `1` to `15`) to rerank the documents. To run this program using the following command:
```
python3 prob_rerank.py query-file top-100-file collection-file expansion-limit [--output resultfile]
```

**positional arguments:**
- `query-file`:         file containing the queries in the same tsv format as
                        given above
- `top-100-file`:       a file containing the top100 documents in the same
                        format as train and dev top100 files given, which need
                        to be reranked
- `collection-file`:    file containing the full document collection (in the
                        same format as msmarco-docs file given)
- `expansion-limit`:    is a number ranging from 1-15 that specifies the limit
                        on the number of additional terms in the expanded
                        query


***optional arguments:***
- `-h`, `--help`:            show this help message and exit
- `--output resultfile`:     the output file named resultfile which is generated by
                             the program after reranking



### `lm_rerank.py`
This program uses *Lavrenko and Croft’s* relevance model using **Unigram Model with Dirichlet Smoothing** (`uni`) and **Bigram Model with Dirichlet Smoothing with Unigram Backoff** (`bi`). To run the program, use:

```
python3 lm_rerank.py query-file top-100-file collection-file model=uni|bi [--output resultfile]
```

**positional arguments:**
- `query-file`:         file containing the queries in the same tsv format as
                        given above
- `top-100-file`:       a file containing the top100 documents in the same
                        format as train and dev top100 files given, which need
                        to be reranked
- `collection-file`:    file containing the full document collection (in the
                        same format as msmarco-docs file given)
- `model`:              it specifies the unigram (`uni`) or the bigram (`bi`)
	                    language model that should be used for relevance language
	                    model *[choose from {"uni","bi"}*

***optional arguments:***
- `-h`, `--help`:            show this help message and exit
- `--output resultfile`:     the output file named resultfile which is generated by
                             the program after reranking

## Footnotes

---
Note: The programs were built for MS-MARCO dataset *after some pre-processing which includes stripping of HTML tags* and might not suit other datasets without proper pre-processing.

*More info and performance report can be obtained from [design.pdf](design.pdf)*


***Implemented by [Subhalingam D](https://subhalingamd.github.io)***
