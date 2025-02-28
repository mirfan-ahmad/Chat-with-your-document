# DocumentQNA - Chat with Your Documents

## Overview
DocumentQNA is an AI-powered system that allows users to interact with their documents using a chatbot interface. It leverages Retrieval-Augmented Generation (RAG) to extract meaningful insights from PDFs, using OpenAI embeddings and LangChain for document retrieval and query response generation.

This project includes both a command-line interface and a Streamlit-based web application for a user-friendly chat experience.

## Features
- Load and process PDF documents.
- Chunk documents using Semantic and Recursive Character Text Splitters.
- Store embeddings in a vector database (ChromaDB).
- Utilize OpenAI's GPT-3.5-turbo for answering document-related queries.
- Context-aware chatbot with a history-aware retriever.
- Streamlit-based interactive UI for document Q&A.

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API Key
- Required libraries: `langchain`, `streamlit`, `chromadb`, `pypdf`, `python-dotenv`, `openai`, `tiktoken`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DocumentQNA.git
   cd DocumentQNA
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Streamlit Web App
To launch the chatbot UI, run:
```bash
streamlit run app.py
```
Upload a PDF and start chatting with your document!

## Project Structure
```
DocumentQNA/
│── qna_chatbot.py       # Core document processing and retrieval logic
│── app.py               # Streamlit web interface
│── main.py              # Command-line interface
│── requirements.txt     # Dependencies
│── .env                 # API keys and environment variables
│── README.md            # Project documentation
│── temp/                # Temporary storage for uploaded files
```

## Technologies Used
- **LangChain**: For document processing and retrieval.
- **OpenAI API**: GPT-3.5-turbo for answering queries.
- **ChromaDB**: Vector database for storing document embeddings.
- **Streamlit**: UI for interactive chatbot experience.
- **Python-dotenv**: Managing environment variables.

## Contributing
Feel free to submit issues or pull requests to improve this project!

## License
This project is licensed under the MIT License.
