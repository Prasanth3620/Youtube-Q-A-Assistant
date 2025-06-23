import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="YouTube RAG Assistant")

st.title("YouTube RAG Assistant")
st.write("Enter a YouTube video ID and your question. The assistant will answer using the transcript of the video.")

# Input fields
video_id = st.text_input("YouTube Video ID (e.g., Gfr50f6ZBvo)")
question = st.text_input("Ask your question")

if st.button("Submit"):
    if not video_id or not question:
        st.warning("Please enter both video ID and a question.")
    else:
        try:
            st.info(" Downloading transcript...")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)

            st.success("Transcript loaded!")

            # Chunking
            st.info("Splitting transcript into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_text(transcript)
            chunk_docs = [Document(page_content=chunk) for chunk in chunks]

            # Embedding + Vector Store
            st.info("Generating embeddings and building vector store...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(chunk_docs, embeddings)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            # Prompt & LLM
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=["context", "question"]
            )

            # RAG Flow
            st.info("Retrieving relevant transcript chunks...")
            retrieved_docs = retriever.invoke(question)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            final_prompt = prompt.invoke({"context": context_text, "question": question})

            st.info("Generating answer...")
            answer = llm.invoke(final_prompt)

            st.success("Answer:")
            st.markdown(answer.content)

        except TranscriptsDisabled:
            st.error("This video has no available transcripts.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
