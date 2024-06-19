def decide_to_search_or_generate(state):
  """
  Determines whether to generate an answer, or do web search

  Args:
      state (dict): The current graph state

  Returns:
      str: Next node to call
  """
  print("---ASSESS GRADED DOCUMENTS---")
  
  state["question"]
  documents = state["documents"]
  
  # documents가 없는 경우 web_search 노드 실행
  if len(documents) == 0:
    print(
        "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
    )
    return "websearch"
  else:
    # We have relevant documents, so generate answer
    print("---DECISION: GENERATE---")
    return "generate"
