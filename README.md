# Youtube-Q-A-Assistant
A RAG-based app that answers questions about YouTube videos using their transcripts. It extracts captions, chunks and embeds them with OpenAI, stores them in FAISS, retrieves relevant context, and uses GPT-4o to generate answers. Built with LangChain and Streamlit for an interactive UI.


# YouTube Transcript Q&A Assistant

This project is a **Retrieval-Augmented Generation (RAG)** application that allows users to ask natural language questions about the content of any **YouTube video** with English captions. The system extracts transcripts using the YouTube Transcript API, splits them into manageable chunks, and generates vector embeddings using OpenAI’s `text-embedding-3-small` model. These embeddings are stored in a FAISS vector database, enabling fast and accurate similarity search. When a user submits a query, the system retrieves the most relevant transcript chunks and uses OpenAI’s GPT-4o model to generate context-aware answers. The application is powered by **LangChain** for orchestration and provides a clean user interface through **Streamlit**.

## Built With

- [LangChain](https://github.com/langchain-ai/langchain) – for prompt orchestration and chaining  
- [OpenAI API](https://platform.openai.com/) – GPT-4o for generation and `text-embedding-3-small` for embeddings  
- [FAISS](https://github.com/facebookresearch/faiss) – fast vector similarity search  
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) – fetches video captions  
- [Streamlit](https://streamlit.io) – simple and interactive user interface  

## Features

- Input any YouTube video ID and ask contextual questions  
- Automatically extracts and processes video transcripts  
- Chunks and embeds transcript text for vector-based retrieval  
- Retrieves top-k relevant segments and feeds them to GPT-4o  
- Interactive Streamlit UI for real-time Q&A  
