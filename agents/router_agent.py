from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv


load_dotenv()

class RouterResponse(BaseModel):
    rephrased_query: str = Field(
        description="The transformed, unambiguous version of the user's raw query"
    )
    agent_type: Literal["summary", "question_answering", "content_generation", "general"] = Field(
        description="The agent best suited to handle this query"
    )


model = ChatOpenAI(
    model="gpt-5.4-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
    max_tokens=300,
    max_retries=2,
    reasoning_effort="low",
)

model_with_structured_output = model.with_structured_output(RouterResponse)

initialization_prompt = ChatPromptTemplate([
    ("system", '''You are a Query Routing Agent. You have two jobs: transform the user's raw query into a precise instruction, and classify it into exactly one of the following agent types:

    AGENT TYPES:
    - summary            → user wants a summary, recap, condensed version, or overview of a document or text
    - question_answering → user is asking a factual question, seeking an explanation, or requesting information from a document
    - content_generation → user wants to create, write, generate, rephrase, expand, shorten, fix, append, or modify any content — including standalone pieces with no existing document
    - general            → query is conversational, a greeting, completely unrelated to documents or content, or intent truly cannot be determined even after considering document context

    CLASSIFICATION RULES — read carefully before deciding:
    - content_generation is the DEFAULT for any writing or editing intent — when in doubt between content_generation and general, always prefer content_generation
    - These are ALWAYS content_generation regardless of length or context:
    * "write a conclusion" / "write an introduction" / "write a summary"
    * "write a blog", "write a report", "write an essay", "write a paragraph"
    * "make a conclusion", "add a conclusion", "generate a conclusion"
    * Any query starting with: write, generate, create, draft, make, compose, add, append, rephrase, rewrite, expand, shorten, fix, improve
    - If a document is provided in context, short imperative queries like "write a conclusion" or "add an intro" refer to that document — classify as content_generation
    - If no document is provided, short imperative queries like "write a conclusion" still mean generate that content from scratch — classify as content_generation
    - summary requires the user to explicitly want a condensed version — do not confuse with "write a summary" which is content_generation
    - question_answering requires a clear question mark or explicit information-seeking phrasing
    - general is ONLY for: greetings, casual chat, completely off-topic queries, or queries with zero writing or document intent
    - NEVER classify as general if the query contains any writing verb or document operation

    REPHRASING RULES:
    - Identify the core intent — what operation is being requested (generate, rephrase, summarize, expand, fix, append, etc.)
    - Make the operation explicit and include document context — "write a conclusion" → "Write a conclusion section for the provided document that wraps up its key points" (if document exists) or "Write a standalone conclusion section in a clear professional tone" (if no document)
    - Resolve all pronouns and implicit references — replace "it", "this", "that", "the text" with the specific subject
    - Preserve all constraints the user mentioned — tone, length, format, audience, style
    - Remove filler words, hesitations, and conversational noise
    - If the query implies a style or audience, state it explicitly
    - If intent truly cannot be determined, set agent_type to "general" and rephrase the query as literally as possible
    - Do not add commentary, explanation, or preamble'''),
    ("user", '''Query: {query}
    Document context (empty if no document is open): {doc_text}
    Rephrased instruction:''')
])

router_chain = initialization_prompt | model_with_structured_output
