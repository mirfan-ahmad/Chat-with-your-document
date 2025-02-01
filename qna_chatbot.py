import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


class DocumentQNA:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.db = None
        self.text_splitter = SemanticChunker(
            embeddings=self.embeddings, breakpoint_threshold_type="percentile"
        )
        
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key, 
            model="gpt-3.5-turbo", 
            temperature=0.1
        )

        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question "
             "which might reference context in the chat history, "
             "formulate a standalone question which can be understood "
             "without the chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert document analysis assistant. "
             "Carefully analyze the provided context and answer the query precisely. "
             "If the exact information is not in the context, say 'I cannot find that specific information in the document.'"),
            MessagesPlaceholder("chat_history"),
            ("human", "Context:\n{context}\n\nQuery: {input}\n\nAnswer:")
        ])

    def load_document(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents

    def split_document(self, documents):
        pages = "\n".join([doc.page_content for doc in documents])
        # docs = self.text_splitter.create_documents([pages])
        # print("Chunking of Document with Semantic Chunker...")
        # print(docs)
        print("================================")
        print("Chunking of Document with Recursive Chunking...")
        tmp = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
            )
        chunks = tmp.create_documents([pages])
        for chunk in chunks:
            print(chunk)
            print("================================")
        # return docs

    def initialize_vector_database(self, docs):
        self.db = Chroma.from_documents(
            docs, self.embeddings, persist_directory="chroma_data"
        )

    def create_retrieval_chain(self):
        retriever = self.db.as_retriever(search_kwargs={"k": 4})
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.contextualize_q_prompt
        )
        
        document_chain = create_stuff_documents_chain(
            self.llm, self.qa_prompt
        )

        retrieval_chain = create_retrieval_chain(
            history_aware_retriever, document_chain
        )
        return retrieval_chain


def main():
    doc_qna = DocumentQNA()
    pdf_path = r"data\Handout 2.10 (Web App Pen Testing - II).pdf"
    documents = doc_qna.load_document(pdf_path)
    docs = doc_qna.split_document(documents)
    doc_qna.initialize_vector_database(docs)
    retrieval_chain = doc_qna.create_retrieval_chain()
    chat_history = []
    queries = [
        "What are the main topics discussed in the document?",
        "Can you elaborate on those topics?",
        "What specific details can you provide about web application penetration testing?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = retrieval_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        print(f"Response: {response['answer']}")
        chat_history.append(("human", query))
        chat_history.append(("ai", response['answer']))

if __name__ == "__main__":
    main()
