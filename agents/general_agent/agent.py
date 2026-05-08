from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class GeneralResponse(BaseModel):
    general_response: str = Field(description="The response to the user's query")

model = ChatOpenAI(
    model="gpt-5.4-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
    max_retries=2,
    reasoning_effort="medium"
)

model_with_structured_output = model.with_structured_output(GeneralResponse)

general_prompt = ChatPromptTemplate([
    ("system", '''You are a Fallback Agent. You receive queries that could not be categorized by the routing system. Your job is to resolve the query as accurately as possible using the provided document, or from general knowledge if no document is given.

    Follow these rules:
    - If a document is provided, answer strictly from its content — do not blend in outside knowledge
    - If the document does not contain enough information, respond exactly with: "The provided document does not contain an answer to this question."
    - If no document is provided, answer from general knowledge in a clear, professional tone
    - Identify what type of response the query needs before answering — factual lookup, explanation, comparison, opinion, instruction, etc.
    - If the query has multiple parts, address each part separately and in order
    - Keep the answer as concise as the question allows — do not over-explain or pad
    - Write in clear, neutral language — no opinions or interpretations beyond what the document or query supports
    - If the query is completely unanswerable even with general knowledge, respond exactly with: "This query cannot be answered with the available information."
    - Output ONLY the answer — no preamble, labels, or closing remarks
    '''),
    ("user", '''Query: {rephrased_query}
    Document: {doc_text}''')
])

general_chain = general_prompt | model_with_structured_output
