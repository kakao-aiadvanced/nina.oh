# https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html#langchain_core.vectorstores.VectorStore.as_retriever

def retrieve_query_and_get_docs(db, query):  
  retriever = db.as_retriever(search_kwargs={"k": 5})  
  search_results = retriever.get_relevant_documents(query)
  return search_results
