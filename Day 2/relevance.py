from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def get_relevance_score(chunk, query):  
  prompt_template_str = """
  너는 주어진 쿼리에서 요구하는 답을 알아낼 수 있는지 평가하는 도구야.
  주어진 본문에 대해서 주어진 쿼리에 해당하는 답이 있는지를 평가해서 정량적인 점수로 나타내야 해.
  주어진 본문을 요약하고 연관도, 유사도를 평가해주고, 0 - 1 사이의 점수로 나타내 줘.
  관련도가 없으면 0으로 나타내 줘.
  Return a JSON object.
  [Example]
  
  

  [본문]
  {document}
  [쿼리]
  {query}
  """

  parser = JsonOutputParser()

  prompt = PromptTemplate(
      template=prompt_template_str,
      input_variables=["document", "query"],
      partial_variables={"format_instructions": parser.get_format_instructions()},
  )

  prompt.format(document=chunk, query=query)
  print(prompt)
  llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key='sk-')
  chain = prompt | llm | parser
  print(chain.invoke({"document": chunk, "query": query}))
  print(chain)
  
  