
from langchain_community.document_loaders import WebBaseLoader
def load_docs():
  urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
  ]
  loader = WebBaseLoader(web_paths=(urls[0], urls[1], urls[2]))
  docs = loader.load()
  file_paths = [
    '/Users/ninaoh/LLM-advanced/Project/data/agent.txt',
    '/Users/ninaoh/LLM-advanced/Project/data/prompt.txt',
    '/Users/ninaoh/LLM-advanced/Project/data/attack.txt'
  ]
  for i in range(3):
    with open(file_paths[i], 'w') as file:
      file.write(docs[i].page_content)
  return file_paths