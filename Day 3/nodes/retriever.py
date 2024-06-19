from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

def retriever():  
  urls = [
      "https://lilianweng.github.io/posts/2023-06-23-agent/",
      "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
      "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
  ]

  docs = [WebBaseLoader(url).load() for url in urls]
  docs_list = [item for sublist in docs for item in sublist]

  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=250, chunk_overlap=0
  )
  doc_splits = text_splitter.split_documents(docs_list)

  # Add to vectorDB
  embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
  vectorstore = Chroma.from_documents(
      documents=doc_splits,
      collection_name="rag-chroma",
      embedding=embedding_function
  )
  return vectorstore.as_retriever()

def retriever_node(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever().invoke(question)
    return {"documents": documents, "question": question}
