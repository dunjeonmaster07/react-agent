import os,sys
from dotenv import load_dotenv
from src.agent import app, print_trace

#sys.path.insert(os.path.join(os.path.dirname(__file__),"src"))
load_dotenv()


chat_history = []
print()
print("Welcome to CLI version of ReAct Agent")
print()

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in ["q","quit","no","exit"]:
        break
    chat_history.append(("user", user_input))
    prev_len = len(chat_history)

    try:
        result = app.invoke({"messages": chat_history})
        chat_history = result["messages"]
        print_trace(result, prev_len)
        print(f"\nAgent: {result['messages'][-1].content}")
    except Exception as e:
        print(f"\n[Error] The agent encountered an issue: {e}")
        print("Try rephrasing your question.")
        chat_history.pop()

print("Goodbye!")