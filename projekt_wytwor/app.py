import streamlit as st
from chatbot_backend import chat_answer

# Set page configuration
st.set_page_config(
    page_title="Team Stress Management Chatbot",
    page_icon="💬",
    layout="centered"
)

# Add header
st.title("Asystent zarządzania stresem w zespole 🤖")
st.markdown("""
Ten chatbot został zaprojektowany, aby pomóc liderom zespołów skutecznie zarządzać stresem i dynamiką zespołu.
Możesz zadawać pytania dotyczące:
- Poziomu stresu w zespole na podstawie wyników ankiet
- Strategii zarządzania zespołem
- Technik radzenia sobie ze stresem
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Co chciałbyś wiedzieć o zarządzaniu stresem w zespole?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Myślę..."):
            response = chat_answer(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response}) 