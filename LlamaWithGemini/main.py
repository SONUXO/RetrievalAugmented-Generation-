import streamlit as st
import os
import pandas as pd
from modules.embedding import create_embedding
from modules.query_engine import query_index

# Streamlit UI
st.title("Chat with Your Documents using Gemini")

# File Upload
st.subheader("Upload Documents")
uploaded_files = st.file_uploader("Upload PDF, TXT, or CSV files", type=["pdf", "txt", "csv"], accept_multiple_files=True)

if uploaded_files:
    # Save files to "Data/" directory
    for uploaded_file in uploaded_files:
        file_path = os.path.join("Data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success("Files uploaded successfully!")

# Button to process documents
if st.button("Process Documents"):
    index = create_embedding()
    if index:
        st.success("Documents processed successfully!")
    else:
        st.error("Error processing documents. Check logs.")

# Query Section
st.subheader("Ask a Question About the Documents")
query = st.text_input("Enter your question")

if query:
    response = query_index(query)
    st.write("**Answer:**", response.response)
