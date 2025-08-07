import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.main import initialize_chatbot
import streamlit as st
from langchain_core.messages import HumanMessage

def handle_input():
    user_input = st.session_state.get("user_input", "")
    if user_input:
        st.session_state['messages'].append({"role": "user", "content": user_input})
        input_data = {"messages": [("user", user_input)]}
        state = {"messages": st.session_state['messages']}
        config = {'configurable': {'thread_id': '1'}}

        try:
            events = chatbot_graph.stream(
                input_data,
                state=state,
                config=config,
                stream_mode='values',
            )
            response = None
            for event in events:
                response = event['messages'][-1]
            if response:
                st.session_state['messages'].append({"role": "ai", "content": response.content})
        except Exception as e:
            st.session_state['messages'].append({"role": "ai", "content": str(e)})
if __name__ == "__main__":
    chatbot_graph = initialize_chatbot()

    st.title("LangGraph Project Graph Visualization Tutorial")
    st.header("Chatbot with Project Graph Visualization")

    if "messages" not in st.session_state:
        st.session_state['messages'] = []

    with st.container():
        for message in st.session_state['messages']:
            if isinstance(message, dict) and message['role']=='ai':
                st.write(f'Bot: {message["content"]}')
            elif isinstance(message, dict) and message['role']=='user':
                st.write(f'You: {message["content"]}')


    # Input Box:
    st.text_input("You: ", key='user_input', on_change=handle_input)




