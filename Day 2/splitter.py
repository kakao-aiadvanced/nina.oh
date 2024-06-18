from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter(file_paths):
  docs = []
  for i in range(len(file_paths)):
    with open(file_paths[i]) as f:
      docs.append(f.read())
  text_splitter = RecursiveCharacterTextSplitter(
      # Set a really small chunk size, just to show.
      chunk_size=100,
      chunk_overlap=50,
      length_function=len,
      is_separator_regex=False,
  )
  return text_splitter.create_documents(docs)