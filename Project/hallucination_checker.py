from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

def hallucination_checker(answer):
  # TODO: 프롬프트를 llama index 표준 형식으로 변경
  prompt_template_str = """
  너는 주어진 본문이 사실인지 여부를 평가하는 도구야.
  본문이 왜 신뢰할만한지 근거를 마련해서 'description' 필드에 적어줘.
  'description'을 기반으로 본문이 사실인 경우, 'hallucination' 필드를 'yes' 로, 아닌 경우 'no'로 채워줘.
  Return a JSON object.
  
  [본문]
  {answer}
  """

  parser = JsonOutputParser()

  prompt = PromptTemplate(
      template=prompt_template_str,
      input_variables=["answer"],
      partial_variables={"format_instructions": parser.get_format_instructions()},
  )

  prompt.format(answer=answer)
  llm = ChatOllama(model='llama3', temperature=0)
  chain = prompt | llm | parser
  return chain.invoke({"answer": answer})
