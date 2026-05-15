import streamlit as st
from src.agent import app, print_trace
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="ReAct Agent", page_icon="🤖", layout="wide")
st.title("reAct Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.write(msg.content)
    elif msg.type == "ai" and not msg.tool_calls:
        with st.chat_message("assistant"):
            st.write(msg.content)

user_input = st.chat_input("Ask me anything...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append(("user", user_input))
    prev_len = len(st.session_state.messages)
    try:
        with st.spinner("Thinking..."):
            result = app.invoke({"messages": st.session_state.messages})
    except Exception as e:
        st.error(f"Agent error: {e}. Try rephrasing your question.")
        st.session_state.messages.pop()
        st.stop()

    st.session_state.messages = result["messages"]
    new_messages = result["messages"][prev_len:]

    tool_steps = []
    final_answer = ""

    for msg in new_messages:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_steps.append(tc)
        elif msg.type == "tool":
            tool_steps.append({"name": "result", "args": msg.content[:300]})
        elif msg.type == "ai" and not msg.tool_calls:
            final_answer = msg.content

    with st.chat_message("assistant"):
        st.write(final_answer)

        if tool_steps:
            with st.expander("Tool trace"):
                for step in tool_steps:
                    name = step["name"]
                    if name == "result":
                        st.caption(f"↩ {step['args']}")
                    elif name == "search_knowledge_base":
                        st.success(f"📚 {name}({step['args']})")
                    elif name == "web_search":
                        st.info(f"🌐 {name}({step['args']})")
                    else:
                        st.info(f"🔧 {name}({step['args']})")
