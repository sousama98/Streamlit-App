import streamlit as st
import os
from dotenv import load_dotenv
from rag import query
from langchain_google_genai import ChatGoogleGenerativeAI 

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_AI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

st.title("ChatBot Application")
chat_placeholder = st.empty()

def init_chat_history():
    """Initialize chat history with a system message."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
def start_chat():
    """Start the chatbot conversation."""
    # Display chat messages from history on app rerun
    with chat_placeholder.container():
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from Chat models
        response = query(prompt)
        
        # message_placeholder.markdown(response)
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})


if __name__ == "__main__":
    init_chat_history()
    start_chat()
