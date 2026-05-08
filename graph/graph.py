import sys
import warnings
from pathlib import Path
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from document_schema.schema import Document, DocumentBody
from typing import Optional

class SynnoiaState(BaseModel):
    query: str = Field(description="The user's query")
    document_name: str = Field(description="The document name to be used for the response")
    doc_text: str = Field(description="The document text to be used for the response")
    doc_json: str = Field(description="The document JSON to be used for the response")
    rephrased_query: str = Field(description="The rephrased query")
    intent: str = Field(description="The intent of the user's query")
    response: str = Field(description="The response to the user's query")
    response_json: Document = Field(description="The response JSON to be used for the response")
    operation_type: str = Field(description="The operation type to be used for the response")
    anchor_id: Optional[str] = Field(default=None, description="The anchor ID to be used for the response")

synnoia_graph = StateGraph(SynnoiaState)

def router_agent(state: SynnoiaState):
    from agents.router_agent import router_chain
    result = router_chain.invoke({"query": state.query,"doc_text":state.doc_text})
    return {
        "rephrased_query": result.rephrased_query,
        "intent": result.agent_type
    }

def summary_agent(state: SynnoiaState):
    from agents.summary_agent.agent import summary_chain
    result = summary_chain.invoke({"rephrased_query": state.rephrased_query, "doc_text": state.doc_text})
    return {
        "response": result.summary,
    }  

def question_answering_agent(state: SynnoiaState):
    from agents.QA_agent.agent import question_answering_chain
    result = question_answering_chain.invoke({"rephrased_query": state.rephrased_query, "doc_text": state.doc_text})
    return {
        "response": result.answer
    }  

def content_generation_agent(state: SynnoiaState):
    from agents.content_generation_agent.agent import content_generation_chain
    result = content_generation_chain.invoke({"rephrased_query": state.rephrased_query, "doc_json": state.doc_json})
    return {
        "response_json": result.document,
        "operation_type": result.operation_type,
        "anchor_id": result.anchor_id
    }  

def general_agent(state: SynnoiaState):
    from agents.general_agent.agent import general_chain
    result = general_chain.invoke({"rephrased_query": state.rephrased_query, "doc_text": state.doc_text})
    return {
        "response": result.general_response
    }

def router(state: SynnoiaState):
    intent = state.intent
    if intent == "summary":
        return "summary_agent"
    elif intent == "question_answering":
        return "question_answering_agent"
    elif intent == "content_generation":
        return "content_generation_agent"
    elif intent == "general":
        return "general_agent"
    else:
        return "general_agent"

synnoia_graph.add_node("router_agent", router_agent)
synnoia_graph.add_node("summary_agent", summary_agent)
synnoia_graph.add_node("question_answering_agent", question_answering_agent)
synnoia_graph.add_node("content_generation_agent", content_generation_agent)
synnoia_graph.add_node("general_agent", general_agent)

synnoia_graph.add_edge(START, "router_agent")
synnoia_graph.add_conditional_edges("router_agent", router,{
    "summary_agent": "summary_agent",
    "question_answering_agent": "question_answering_agent",
    "content_generation_agent": "content_generation_agent",
    "general_agent": "general_agent"
})
synnoia_graph.add_edge("summary_agent", END)
synnoia_graph.add_edge("question_answering_agent", END)
synnoia_graph.add_edge("content_generation_agent", END)
synnoia_graph.add_edge("general_agent", END)

synnoia_agent = synnoia_graph.compile()


