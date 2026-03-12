"""
Entry point for the Berkshire transcript RAG conversational agent.
"""

from vectorstore import load_or_build_vectorstore
from rag_chain import build_rag_chain


def main():
    print("Loading vector store...")
    store = load_or_build_vectorstore("documents")

    print("Building RAG chain...")
    chain = build_rag_chain(store)

    print("\n" + "=" * 60)
    print("  Berkshire Meeting Q&A Agent")
    print("  Ask anything about Buffett & Munger's wisdom.")
    print("  Type 'quit' or 'exit' to leave.")
    print("=" * 60 + "\n")

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


if __name__ == "__main__":
    main()
