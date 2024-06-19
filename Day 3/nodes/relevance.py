from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

def relevance():
  llm = ChatOllama(model='llama3', temperature=0)

  prompt = PromptTemplate(
      template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
      of a retrieved document to a user question. If the document contains keywords related to the user question,
      grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
      Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
      Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
       <|eot_id|><|start_header_id|>user<|end_header_id|>
      Here is the retrieved document: \n\n {document} \n\n
      Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
      """,
      input_variables=["question", "document"],
  )

  return prompt | llm | JsonOutputParser()

def relevance_node(state):
  """
  Determines whether the retrieved documents are relevant to the question
  If any document is not relevant, we will set a flag to run web search

  Args:
      state (dict): The current graph state

  Returns:
      state (dict): Filtered out irrelevant documents
  """

  print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
  question = state["question"]
  documents = state["documents"]

  # Score each doc
  filtered_docs = []
  for d in documents:
    score = relevance().invoke(
        {"question": question, "document": d.page_content}
    )
    grade = score["score"]
    # Document relevant
    if grade.lower() == "yes":
        print("---GRADE: DOCUMENT RELEVANT---")
        filtered_docs.append(d)
    # Document not relevant
    else:
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        # We do not include the document in filtered_docs
        # We set a flag to indicate that we want to run web search
        continue
  return {"documents": filtered_docs, "question": question}

