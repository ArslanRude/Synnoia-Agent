from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class SummaryOutput(BaseModel):
    summary: str = Field(..., description="The summarized text")

model = ChatOpenAI(
    model="gpt-5.4-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
    max_tokens=5000,
    max_retries=2,
    reasoning_effort="low"
)

model_with_structured_output = model.with_structured_output(SummaryOutput)

summary_prompt = ChatPromptTemplate([
    ("system", '''You are a Summarization Agent. Your job is to read the provided text and produce a concise, accurate summary.
    Follow these rules:
    - Preserve all key facts, names, dates, and figures — never omit critical information
    - Use clear, neutral language — no opinions or interpretations not present in the original
    - Write in third person, present tense
    - Scale the summary length proportionally to the source text — short texts get 1-2 sentences, long or complex texts get more, but always be as brief as the content allows
    - Output ONLY the summary — no preamble, labels, or closing remarks
    '''),
    ("user", '''Query: {rephrased_query}
    Document: {doc_text}''')
])

summary_chain = summary_prompt | model_with_structured_output
