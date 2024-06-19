from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

def hallucination():
  llm = ChatOllama(model='llama3', temperature=0)

  # Prompt
  prompt = PromptTemplate(
      template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
      an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
      whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
      single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
      Here are the facts:
      \n ------- \n
      {documents}
      \n ------- \n
      Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
      input_variables=["generation", "documents"],
  )

  return prompt | llm | JsonOutputParser()


def hallucination_node(state):
  """
  Determines whether the generated output has hallucination from documents
  If generated output has any hallucination, we will regenerate an answer.

  Args:
      state (dict): The current graph state

  Returns:
      state (dict): Remove generation includes hallucination
  """

  print("---CHECK GENERATION HAS HALLUCINATION---")

  question = state["question"]
  documents = state["documents"]
  generation = state["generation"]

  docs_str = ""
  for i in range(len(documents)):
    docs_str += documents[i].page_content
  
  score = hallucination().invoke(
      {"generation": generation, "documents": docs_str}
  )
  grade = score["score"]

  if grade.lower() == "no":
      print("---CHECK: GENERATION HAS HALLUCINATION---")
      # Document not relevant
      return {"question": question, "generation": "", "documents": documents}
  else:
      print("---CHECK: GENERATION HAS NO HALLUCINATION---")
      # We do not include the document in filtered_docs
      # We set a flag to indicate that we want to run web search
      return {"question": question, "generation": generation, "documents": documents}

