#!/usr/bin/env python
# coding: utf-8

"""
PDF Summarizer using LangChain, FAISS, HuggingFace Embeddings, and GPT-3.5 Turbo
"""

# === Imports ===
import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

# === Streamlit Page Config ===
st.set_page_config(page_title="PDF Summarizer", layout="centered")
st.title("📄 PDF Summarizer")
st.markdown("Upload a PDF and get a concise 3–5 line summary using OpenAI GPT-3.5 and LangChain.")

# === PDF Upload ===
uploaded_pdf = st.file_uploader("📤 Upload your PDF file", type="pdf")


# === Text Processing Function ===
def process_text(text: str):
    """
    Splits the input text into chunks and creates a FAISS vector store using embeddings.

    Parameters:
        text (str): The extracted full text from the PDF.

    Returns:
        FAISS: A FAISS vector store for similarity search.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store


# === Summarization Function ===
def generate_summary(pdf_file):
    """
    Extracts text from the uploaded PDF, builds a knowledge base, and queries a summary.

    Parameters:
        pdf_file: Uploaded PDF file from Streamlit uploader.

    Returns:
        str: The summarized content.
    """
    try:
        pdf_reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

        if not text.strip():
            return "No text content found in the PDF."

        vector_store = process_text(text)
        query = "Summarize the content of the pdf in 3-5 lines"
        relevant_docs = vector_store.similarity_search(query)

        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.8)
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as callback:
            response = chain.run(input_documents=relevant_docs, question=query)
            print(f"Token usage and cost: {callback}")
            return response

    except Exception as e:
        return f"❌ Error: {str(e)}"


# === Streamlit UI Logic ===
if uploaded_pdf is not None:
    with st.spinner("🔍 Summarizing the document..."):
        summary = generate_summary(uploaded_pdf)

    if summary:
        st.subheader("📋 Summary")
        st.write(summary)
    else:
        st.error("⚠️ Unable to generate summary. Please check the PDF content.")
