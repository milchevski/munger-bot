"""
Entry point for the Berkshire transcript RAG conversational agent.
"""

import os
from datetime import datetime

from vectorstore import load_or_build_vectorstore
from rag_chain import build_rag_chain

_LOG_FILE = "conversation_log.md"


def _append_to_log(question: str, answer: str, is_new_session: bool = False) -> None:
    """Append a Q&A exchange to the markdown log file."""
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        if is_new_session:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n---\n\n## Session — {timestamp}\n\n")
        f.write(f"### Q: {question}\n\n{answer}\n\n")


def main():
    print("Loading vector store...")
    store = load_or_build_vectorstore("documents")

    print("Building RAG chain...")
    chain = build_rag_chain(store)

    # Create log file with a title if it doesn't exist yet
    if not os.path.exists(_LOG_FILE):
        with open(_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("# Berkshire Meeting Q&A — Conversation Log\n")

    print("\n" + "=" * 60)
    print("  Berkshire Meeting Q&A Agent")
    print("  Ask anything about Buffett & Munger's wisdom.")
    print("  Type 'quit' or 'exit' to leave.")
    print(f"  Responses are saved to {_LOG_FILE}")
    print("=" * 60 + "\n")

    is_new_session = True

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        response = chain.invoke(question)
        print(f"\nAgent: {response}\n")

        _append_to_log(question, response, is_new_session=is_new_session)
        is_new_session = False


if __name__ == "__main__":
    main()
