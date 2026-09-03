RAG-Powered AI Estimator

RAG-Powered AI Estimator is an AI-based estimation tool that uses Retrieval-Augmented Generation (RAG) to generate more context-aware and reliable estimates from a provided knowledge base. Instead of relying solely on an LLM’s general knowledge, the system first retrieves relevant information from stored documents and then provides that context to the language model to generate an estimate.

The project uses a vector store and embeddings to perform semantic search. When a user submits a query, the system converts the query into an embedding, retrieves the most relevant document chunks, and passes the retrieved context along with the user’s question to the LLM. The generated response is then displayed through the application’s interface.

This project demonstrates practical implementation of RAG, semantic search, vector embeddings, document retrieval, prompt engineering, LLM integration, and a user-facing AI application.

How to Run the Project

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Open the project

cd RAG-Powered-AI-Estimator

Open the folder in VS Code.

3. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

macOS/Linux:

python3 -m venv venv
source venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

5. Configure environment variables

Create a .env file in the project directory and add the required API key(s), for example:

OPENAI_API_KEY=your_api_key_here

Do not upload your .env file or API keys to GitHub.

6. Build the vector store

Run the project’s vector-store creation script:

python build_vectorstore.py

This processes the source documents, creates embeddings, and stores them for retrieval.

7. Start the application

streamlit run app.py

8. Use the application

Streamlit will provide a local URL, typically:

http://localhost:8501

Open it in your browser, enter your estimation-related query, and the application will retrieve relevant information from the knowledge base before generating the final AI-powered estimate.
##

Demo

![RAG Search Engine Demo](Screenshot%202026-09-04%20014745.png)
