
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def embedding(text):
  return embeddings.embed_query(text)
