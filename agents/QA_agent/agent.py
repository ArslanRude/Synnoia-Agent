from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class QAResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")

model = ChatOpenAI(
    model="gpt-5.4-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
    max_tokens=3000,
    max_retries=2,
    reasoning_effort="medium"
)

model_with_structured_output = model.with_structured_output(QAResponse)

QA_prompt = ChatPromptTemplate([
    ("system", '''You are a Question-Answering Agent. Your job is to answer the user's question strictly using the provided document.

    Follow these rules:
    - Answer only from information explicitly stated in the document — do not infer, assume, or use outside knowledge
    - If the document does not contain enough information to answer, respond exactly with: "The provided document does not contain an answer to this question."
    - Keep the answer as concise as the question allows — do not over-explain
    - If the question has multiple parts, address each part separately in order
    - Write in clear, neutral language — no opinions or interpretations not present in the document
    - Output ONLY the answer — no preamble, labels, or closing remarks
    '''),
    ("user", '''Question: {rephrased_query}
    Document: {doc_text}''')
])


question_answering_chain = QA_prompt | model_with_structured_output
