from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

def save_vector(docs):
  embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
  db = Chroma.from_documents(docs, embedding_function)
  return db
