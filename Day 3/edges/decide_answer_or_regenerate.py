def decide_to_answer_or_regenerate(state):
  """
  Determines whether to regenerate an answer, or END

  Args:
      state (dict): The current graph state

  Returns:
      str: Next node to call
  """
  print("---ASSESS GRADED DOCUMENTS---")
  
  generation = state["generation"]
  
  # documents가 없는 경우 web_search 노드 실행
  if generation == "":
    print(
        "---DECISION: GENERATION HAS HALLUCINATION, TRYING TO REGENERATE---"
    )
    return "generate"
  else:
    # We have relevant documents, so generate answer
    print("---DECISION: GENERATION HAS NO HALLUCINATION, END TASK---")
    return "useful"
