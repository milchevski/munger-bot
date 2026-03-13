"""
RAG chain module for the Berkshire transcript conversational agent.

Composes a retrieval-augmented generation chain using:
- InMemoryVectorStore retriever (k=5)
- ChatPromptTemplate with Buffett/Munger persona
- Gemini (gemini-2.0-flash) as the chat model
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

_SYSTEM_PROMPT = """\
You are a knowledgeable assistant that answers questions about investing, \
business, and life wisdom by drawing on the insights of Warren Buffett and \
Charlie Munger from Berkshire Hathaway annual meeting transcripts (1994–2019).

Below are relevant Q&A excerpts retrieved from the transcripts. Each excerpt \
includes the meeting year, the topic/question title, and responses from both \
Buffett and Munger.

{context}

Instructions:
- Ground your answer in the retrieved excerpts above.
- Attribute views to Buffett or Munger specifically when they differ.
- If the retrieved excerpts don't contain enough information, say so honestly.
- Be conversational and insightful, matching the tone of a Berkshire meeting."""

_USER_PROMPT = "{question}"


def _format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a readable context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(
            f"--- Excerpt {i} [{meta['year']}] ---\n"
            f"Topic: {doc.page_content}\n\n"
            f"Warren Buffett: {meta['buffett_answer']}\n\n"
            f"Charlie Munger: {meta['munger_answer']}"
        )
    return "\n\n".join(parts)


def build_rag_chain(vectorstore: InMemoryVectorStore):
    """
    Build an LCEL RAG chain from the given vector store.

    Args:
        vectorstore: A populated InMemoryVectorStore with transcript Q&A docs.

    Returns:
        A runnable chain that accepts {"question": str} and returns a string.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", _USER_PROMPT),
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        max_retries=0,
    )

    def _print_prompt(p):
        print("\n" + "="*40 + " PROMPT TO LLM " + "="*40)
        print(p.to_string() if hasattr(p, 'to_string') else p)
        print("="*95 + "\n")
        return p

    chain = (
        {
            "context": retriever | _format_context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | RunnableLambda(_print_prompt)
        | llm
        | StrOutputParser()
    )

    return chain
