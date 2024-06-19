from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

def generate():
  # Prompt
  prompt = PromptTemplate(
      template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
      Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
      Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
      Question: {question}
      Context: {context}
      Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
      input_variables=["question", "document"],
  )

  llm = ChatOllama(model='llama3', temperature=0)

  return prompt | llm | StrOutputParser()

def generate_node(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = generate().invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
