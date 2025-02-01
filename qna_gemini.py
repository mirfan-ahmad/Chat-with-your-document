import os
from typing import List
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class RAGSystem:
    def __init__(self, google_api_key: str, docs_dir: str = "documents/"):
        self.google_api_key = google_api_key
        self.docs_dir = docs_dir
        self.vectorstore = None
        self.qa_chain = None
        
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=google_api_key,
            temperature=0.3
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )

    def load_documents(self) -> List:
        loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.pdf",
            show_progress=True
        )
        documents = loader.load()
        return documents

    def process_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        documents = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        texts = text_splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="chroma_db"
        )
        self.vectorstore.persist()

    def setup_qa_chain(self):
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, just say so.
        Try to provide detailed answers when possible.
        Context: {context}
        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

    def query(self, question: str) -> dict:
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Run setup_qa_chain() first.")
            
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }


def main():
    google_api_key = os.getenv("GEMINI_API_KEY")
    rag = RAGSystem(google_api_key)
    rag.process_documents()
    rag.setup_qa_chain()
    question = "What is the main topic discussed in the documents?"
    result = rag.query(question)
    
    print("Answer:", result["answer"])
    print("\nSource Documents:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\nDocument {i}:")
        print(doc.page_content[:200] + "...")

if __name__ == "__main__":
    main()