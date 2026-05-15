import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
CHROMA_DIR = os.path.join(os.path.dirname(__file__),"..","chroma_db")
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persisted_chroma_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings) #load chroma d

#RAG knowledge base setup:
@tool
def search_knowledge_base(query: str):
    """Search the knowledge base for Python, RAG, and AI/ML related concepts."""
    retriever = persisted_chroma_db.as_retriever(search_kwargs={"k":3})
    docs = retriever.invoke(query)
    results = []
    for doc in docs:
        results.append(f"{doc.page_content[:250]}\nSource: {doc.metadata}")
    return "\n---\n".join(results)

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b



#Tavily Setup:
#web_search = TavilySearch(max_results=1, include_raw_content=False)
@tool
def web_search(query: str):
    """Search the web for real-time information."""
    tavily = TavilySearch(max_results = 1, include_raw_content = False)
    result = tavily.invoke({"query": query})

    outputs = []
    for r in result["results"]:
        outputs.append(f"{r['title']}\n{r['content']}")
    
    return "\n---\n".join(outputs)










#Use the below for standalone testing:


# result = tavily_tool.invoke({"query": "What happened at the last wimbledon?"})

# print(f"\n{'='*60}")
# print(f"  Query: {result['query']}")
# print(f"  Response Time: {result['response_time']:.2f}s")
# print(f"  Results: {len(result['results'])}")
# print(f"{'='*60}\n")

# for i, r in enumerate(result["results"], 1):
#     print(f"  [{i}] {r['title']}")
#     print(f"      Score: {r['score']:.4f}")
#     print(f"      URL:   {r['url']}")
#     print(f"      {'─'*50}")
#     print(f"      {r['content'][:300]}...")
#     print()


