from load import load_docs
from splitter import splitter
from vector import save_vector
from retriever import retrieve_query_and_get_docs

paths = load_docs()
splitted_docs = splitter(paths)
vector_db = save_vector(splitted_docs)
docs = retrieve_query_and_get_docs(vector_db, 'agent memory')

from relevance import get_relevance_score
for i in range(5):
  relevant_answer = get_relevance_score(docs[i].page_content, 'agent memory')
  print(relevant_answer)
  print(relevant_answer['relevance'])