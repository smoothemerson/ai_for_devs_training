import streamlit as st
from ollama import chat
from ollama import ChatResponse

st.title("Ollama Like")

if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "gpt-oss:20b"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    instructions = "Responda as perguntas do usuário de maneira informal"

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        chat_message: ChatResponse = chat(
            model=st.session_state["ollama_model"],
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
        )

        response = chat_message.message.content
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
