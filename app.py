import streamlit as st
from qna_chatbot import DocumentQNA
import time
import os

@st.cache_resource
class RAGSystem:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        self.doc_qna = DocumentQNA()
        self.retrieval_chain = None
        self.chat_history = []

    def load_user_document(self, pdf_path):
        # Load and split document
        documents = self.doc_qna.load_document(pdf_path)
        docs = self.doc_qna.split_document(documents)
        
        # Initialize vector database
        self.doc_qna.initialize_vector_database(docs)
        
        # Create retrieval chain
        self.retrieval_chain = self.doc_qna.create_retrieval_chain()

    def get_answer(self, query):
        if not self.retrieval_chain:
            raise ValueError("No document loaded. Please upload a document first.")
        
        # Invoke retrieval chain with query and chat history
        response = self.retrieval_chain.invoke({
            "input": query,
            "chat_history": self.chat_history
        })
        
        # Update chat history
        self.chat_history.append(("human", query))
        self.chat_history.append(("ai", response['answer']))
        
        return response['answer']


def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


rag_system = RAGSystem()

# Sidebar configuration
with st.sidebar:
    st.header("Chatbot Settings")
    st.markdown("Configure the chatbot's behavior and appearance.")
    temperature = st.slider("Response Temperature", 0.0, 1.0, 0.5, 0.1)
    rag_system.temperature = temperature
    st.write("Temperature controls the creativity of the responses.")
    st.markdown("---")

    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file is not None and "document_loaded" not in st.session_state:
        with st.spinner("Loading document..."):
            try:
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Load document only once
                rag_system.load_user_document(temp_file_path)
                st.session_state.document_loaded = True
                st.success("Document loaded successfully!")
            except Exception as e:
                st.error(f"Error loading document: {str(e)}")

st.title("Chat with your Document")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input
if prompt := st.chat_input("Ask something about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                # Ensure the document is loaded
                if "document_loaded" not in st.session_state:
                    raise ValueError("No document loaded. Please upload a document first.")

                answer = rag_system.get_answer(prompt)
                formatted_answer = f"{answer}"
                st.session_state.messages.append({"role": "assistant", "content": formatted_answer})

                response_text = "".join(response_generator(answer))
                st.markdown(formatted_answer, unsafe_allow_html=True)
            except Exception as e:
                error_message = f"<div style='color: red;'>Sorry, I encountered an error: {str(e)}</div>"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.markdown(error_message, unsafe_allow_html=True)