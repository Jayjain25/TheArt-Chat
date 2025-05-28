import streamlit as st
from langchain_community.document_loaders import PyPDFLoader # Using pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain.embeddings.base import Embeddings # Type hint
from io import BytesIO
import os
import tempfile

def process_pdfs(uploaded_files, embedding_function):
    """
    Loads PDFs, splits them into chunks, creates embeddings, and builds a FAISS vector store.
    """
    if not uploaded_files:
        st.warning("No PDF files uploaded.")
        return None
    if embedding_function is None:
        st.error("Embedding function not available. Cannot process PDFs.")
        return None

    all_docs = []
    with st.spinner(f"Processing {len(uploaded_files)} PDF file(s)..."):
        for uploaded_file in uploaded_files:
            try:
                # PyPDFLoader needs a file path. Save temp file.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    tmp_file_path = tmpfile.name

                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load() # Loads pages as documents
                # Add source metadata
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["page"] = doc.metadata.get("page", "?") + 1 # page is 0-indexed
                all_docs.extend(docs)

                os.remove(tmp_file_path) # Clean up temp file

            except Exception as e:
                st.error(f"Error processing file '{uploaded_file.name}': {e}")
                # Optionally remove temp file if it still exists
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                     os.remove(tmp_file_path)
                continue # Skip to next file

            if not all_docs:
                 st.warning("No text could be extracted from the uploaded PDF(s).")
                 return None

            # 2. Split Documents into Chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                add_start_index=True, # Add start index to metadata
            )
            split_docs = text_splitter.split_documents(all_docs)

            if not split_docs:
                 st.warning("Text extracted, but failed to split into chunks.")
                 return None

            # 3. Create Embeddings and Vector Store
            try:
                st.info(f"Creating embeddings for {len(split_docs)} text chunks...")
                vector_store = FAISS.from_documents(split_docs, embedding_function)
                st.success(f"PDF(s) processed. Vector store created with {vector_store.index.ntotal} entries.")
                return vector_store
            except Exception as e:
                st.error(f"Error creating vector store: {e}")
                return None

def get_relevant_context(query, vector_store, k=4):
    """
    Performs similarity search on the vector store to find relevant document chunks.
    """
    if vector_store is None:
        return "Error: Vector store not initialized."

    try:
        # Add score threshold? fetch_k?
        relevant_docs = vector_store.similarity_search(query, k=k)
        if not relevant_docs:
            return "No relevant context found in the documents."

        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        # Add sources to the context
        sources = set()
        for doc in relevant_docs:
             source = doc.metadata.get('source', 'Unknown')
             page = doc.metadata.get('page', '?')
             sources.add(f"{source} (Page {page})")
        if sources:
            context += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sorted(list(sources)))

        return context
    except Exception as e:
        st.error(f"Error performing similarity search: {e}")
        return f"Error retrieving context: {e}"