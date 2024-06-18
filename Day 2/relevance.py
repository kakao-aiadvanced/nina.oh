from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

def get_relevance_score(chunk, query):  
  # TODO: 프롬프트를 llama index 표준 형식으로 변경
  prompt_template_str = """
  너는 주어진 쿼리에서 요구하는 답을 알아낼 수 있는지 평가하는 도구야.
  주어진 본문에 대해서 주어진 쿼리에 해당하는 답이 있는지를 평가해서 정량적인 점수로 나타내야 해.
  주어진 본문을 요약하고 연관도, 유사도를 평가해주고, 'score' 필드를 0 - 1 사이의 점수로 나타내 줘. 관련도가 없으면 0으로 나타내 줘
  그리고 연관도가 0.70 이상인 경우, ‘relevance’ 필드를 'yes' 로, 아닌 경우 'no'로 채워줘.
  Return a JSON object.
  
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
  llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key='sk-')
  chain = prompt | llm | parser
  return chain.invoke({"document": chunk, "query": query})

  
  