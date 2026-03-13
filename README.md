# munger-bot
A RAG conversational agent

*Note: This is a hobby project.*

## Parser
The transcripts used in this project were downloaded from [r/ValueInvesting: Berkshire meeting transcripts from 1994-2022](https://www.reddit.com/r/ValueInvesting/comments/wxawlo/berkshire_meeting_transcripts_from_1994_2022/).

The parser processes the raw transcripts and extracts Q&A pairs. It separates the content by year and topic/question title, systematically structuring the answers into distinct responses from Warren Buffett and Charlie Munger, and saving the results into JSON files.

## RAG Agent
The Retrieval-Augmented Generation (RAG) agent is built using **LangChain** and **Google Gemini** (`gemini-2.5-flash`).

It utilizes an `InMemoryVectorStore` populated with embeddings of the parsed transcript Q&A documents. When a user asks a question, the vector store retrieves the 5 most relevant excerpts. The agent then uses a specialized Chat Prompt Template to craft answers, imitating the persona of Warren Buffett and Charlie Munger. The responses are grounded entirely in the retrieved excerpts, specifically attributing differing views to Buffett and Munger, and matching the distinct, conversational tone of a Berkshire Hathaway annual meeting.

All Q&A exchanges are automatically saved to `conversation_log.md` for reference.

## Usage

1. Set up your Gemini API key:
   ```bash
   export GOOGLE_API_KEY="your-api-key"
   ```

2. Install dependencies and run:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install langchain-core langchain-google-genai
   python main.py
   ```
