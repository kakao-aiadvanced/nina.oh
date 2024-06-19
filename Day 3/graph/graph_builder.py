from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph

from nodes.retriever import retriever_node
from nodes.generate import generate_node
from nodes.relevance import relevance_node
from nodes.websearch import web_search_node
from nodes.hallucination import hallucination_node

from edges.decide_to_search_or_generate import decide_to_search_or_generate
from edges.decide_answer_or_regenerate import decide_to_answer_or_regenerate

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        hallucination_cnt: count of running hallucination
    """

    question: str
    generation: str
    documents: List[str]

def generate_workflow():
    workflow = StateGraph(GraphState)
    # Define the nodes
    workflow.add_node("websearch", web_search_node)  # web search
    workflow.add_node("retrieve", retriever_node)  # retrieve
    workflow.add_node("relevance", relevance_node)  # grade documents
    workflow.add_node("generate", generate_node)  # generatae
    workflow.add_node("hallucination", hallucination_node)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "relevance")
    workflow.add_conditional_edges(
        "relevance",
        decide_to_search_or_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "relevance")
    workflow.add_edge("generate", "hallucination")
    workflow.add_conditional_edges(
        "hallucination",
        decide_to_answer_or_regenerate,
        {
            "generate": "generate",
            "useful": END
        },
    )
    return workflow
