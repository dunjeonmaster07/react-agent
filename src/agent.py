from typing import Annotated #A Python type hinting mechanism used to add metadata to types.
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages #A reducer function that instructs the graph to merge new messages into the existing message
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv
from src.tools import add,multiply,web_search,search_knowledge_base

load_dotenv()


'''
Instead of this (basic LLM):

User → Prompt → LLM → Answer

You build this using Langraph:

User → Understand → Decide → Act → Check → Respond
                  ↘ (loop if needed)
'''


#state: Use Annotated[list, add_messages]  Without it, every update to the state would overwrite previous messages. This ensures the model has access to the full conversation context. It automatically manages different message roles (human, AI, tool, system).It is the foundation for creating short-term memory within AgentState
class MessageState(TypedDict):
    # This ensures that whenever a node returns a message,
    # it's appended to the existing list.
    messages : Annotated[list, add_messages]


#nodes: just like lambdas, A node is essentially a "worker" that executes a specific task, such as calling an LLM, querying a database, or performing data transformation.
#Every node receives the current Graph State (and optionally a configuration object) as its input and returns a dictionary representing the updates it wants to make to that state.
# LLM node: Often called "Agent Nodes," these use a language model to reason about the state and decide on the next response or action.
# tool node: Pre-built nodes (like ToolNode) designed specifically to execute external functions or APIs called by the model.
#every node must return a state update dictionary. 
# --- LLM setup ---
# Plain LLM can only respond with text. bind_tools() gives it awareness of
# available tools so its responses can include structured tool call requests.
# This is done once at module level — not per call — to avoid repeated work.
llm = ChatGroq(
    model = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-120b"),
    api_key = os.environ["GROQ_API_KEY"]
)

tools = [add,multiply,web_search,search_knowledge_base]
# for t in tools:
#     print(type(t), getattr(t, "name", "NO_NAME"))
llm_with_tools = llm.bind_tools(tools)


# --- Node 1: Agent (the "brain") ---
# Passes the full conversation history to the LLM, which either:
#   a) responds with text (done thinking), or
#   b) responds with tool_calls (needs more info → triggers the ReAct loop)
def agent_node(state: MessageState):
    """You are a helpful assistant. Use the provided tools when needed. Always use the correct tool calling format."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# --- Node 2: Tool executor ---
# ToolNode is prebuilt — it reads the tool_calls from the last AI message,
# executes the matching function, and returns the result as a ToolMessage.
# No custom logic needed; LangGraph handles the dispatch.
tool_node = ToolNode(tools)


# --- Edges ---
# Edges define the structure, flow, and decision-making logic of the graph,
# acting as the connectors ("roads") between nodes ("rooms" or actions).
# This is the ReAct decision point: does the agent need a tool, or is it done?
# The return value must match a node name ("tools") or END.
def conditional_edge(state : MessageState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool"
    return END


# --- Graph (TODO: wire it up) ---
graph = StateGraph(MessageState) #You need to tell it what shape the data will be — that's your MessageState. knows "my state has a messages list." No nodes, no edges yet — just the container.

#Register nodes:
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

#When a user message arrives, which node should run first? The agent needs to see the message and think.
# Entry
graph.add_edge(START, "agent")
#graph.set_entry_point("agent")

#Decision
graph.add_conditional_edges("agent", conditional_edge)

#loop back
graph.add_edge("tool", "agent")

#Testing
app = graph.compile()


def print_trace(result, start_from=0):
    """Walk through messages from this turn only and print a readable timeline."""
    messages = result["messages"][start_from:]
    print(f"\n{'='*60}")
    for i, msg in enumerate(messages):
        role = msg.type  # "human", "ai", or "tool"

        if role == "human":
            print(f"  Step {i} [USER]:  {msg.content}")

        elif role == "ai":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  Step {i} [AI → tool call]:  {tc['name']}({tc['args']})")
            else:
                print(f"  Step {i} [AI → answer]:  {msg.content[:300]}")

        elif role == "tool":
            print(f"  Step {i} [TOOL result]:  {msg.content[:200]}")

        print(f"  {'─'*56}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    result = app.invoke({"messages": [("user", "What's 15 * 127?")]})
    print_trace(result)
    # ... rest of test calls
    result = app.invoke({"messages": [("user", "What's 15 * 127?")]})
    print_trace(result)

    result = app.invoke({"messages": [("user", "What's 15 + 127?")]})
    print_trace(result)

    result = app.invoke({"messages": [("user", "Explain Python for loop?")]})
    print_trace(result)