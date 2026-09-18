# ReAct Agent — 67 Lines of Python

A working ReAct (Reason + Act) agent built with LangGraph and Groq. The entire agent logic fits in 67 lines of code.

## What it does

The agent receives a question, **thinks** about it, **calls tools** if needed, and **loops** until it has enough information to answer. This is the ReAct pattern — the foundation of most production AI agents.

**4 tools included:**
- `web_search` — real-time web search via Tavily
- `search_knowledge_base` — RAG over local PDFs using ChromaDB + HuggingFace embeddings
- `add` / `multiply` — basic math (demonstrates structured tool calling)

## Architecture

```
User → Agent (LLM thinks) → Tool call? → Yes → Execute tool → Loop back
                                        → No  → Return answer
```

Three building blocks:
1. **State** — `Annotated[list, add_messages]` keeps full conversation memory
2. **Nodes** — Agent node (LLM) + Tool node (executor)
3. **Edges** — Conditional edge decides: call another tool or finish

## Quick start

```bash
git clone https://github.com/dunjeonmaster07/react-agent.git
cd react-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

You need free accounts for:
- [Groq](https://console.groq.com) — LLM (llama-3.3-70b, free tier)
- [Tavily](https://tavily.com) — web search (1,000 free credits/month)
- [LangSmith](https://smith.langchain.com) — tracing (5,000 free traces/month, optional)

### Build the knowledge base (one-time)

The `search_knowledge_base` tool reads from a local Chroma store, which isn't
included in the repo. Embed the PDFs in `data/` into it first:

```bash
python src/ingest.py
```

### Run the CLI

```bash
python main.py
```

### Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Project structure

```
├── src/
│   ├── agent.py          # The 67-line ReAct agent (State, Nodes, Edges, Graph)
│   ├── tools.py           # 4 tools: web search, RAG, add, multiply
│   ├── ingest.py          # One-time: embeds data/*.pdf into the Chroma store
│   └── __init__.py
├── data/                  # PDFs for the RAG knowledge base
├── main.py                # CLI chat interface
├── streamlit_app.py       # Streamlit web UI with tool trace viewer
├── requirements.txt
├── .env.example
└── README.md
```

## Tech stack

| Component | Tool | Cost |
|---|---|---|
| LLM | Groq (llama-3.3-70b-versatile) | Free |
| Agent framework | LangGraph | Free |
| Web search | Tavily | Free (1k/month) |
| Vector DB | ChromaDB (local) | Free |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) | Free |
| Tracing | LangSmith | Free (5k traces/month) |
| UI | Streamlit | Free |

**Total cost to run: $0**

## Built with

Built as part of my AI engineering learning journey. Follow along:
- [LinkedIn](https://www.linkedin.com/in/ankit-chaudhary007)
- [YouTube](https://www.youtube.com/@CodeAgents_ai)
- [Instagram](https://www.instagram.com/codeagents_ai)
- [Newsletter](https://code-agent-ai.beehiiv.com)

## License

MIT
