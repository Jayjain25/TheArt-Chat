import streamlit as st
from langchain_openai import ChatOpenAI # Use OpenAI client for Groq

GROQ_API_BASE = "https://api.groq.com/openai/v1"

# --- Model Listing ---
def get_available_models(api_key):
    if not api_key: return ["Enter API Key"]
    # Models available via Groq (check their console/docs for updates)
    return [ "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it" ]

# --- LLM Instantiation ---
def get_llm(api_key, model, params):
    if not api_key: raise ValueError("Groq API Key missing.")
    try:
        openai_params = {
            "temperature": params.get('temperature', 0.7),
            "max_tokens": params.get('max_tokens', 1024),
        }
        llm = ChatOpenAI(
            model=model,
            api_key=api_key, # Use api_key
            base_url=GROQ_API_BASE, # Use base_url
            **openai_params
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM ({model}): {e}")
        raise

# --- Embeddings ---
def get_embeddings(api_key):
    # Groq API focuses on LLM inference, not embedding models currently.
    st.info("Groq does not provide embedding models via its API. PDF Q&A disabled for Groq.")
    return None

# --- Direct Generation (Optional) ---
def generate_response(api_key, model, messages, params):
    try:
        llm = get_llm(api_key, model, params)
        lc_messages = [msg.to_langchain() for msg in messages]
        response = llm.invoke(lc_messages)
        return response.content
    except Exception as e:
        st.error(f"Groq API Error ({model}): {e}")
        return f"Error generating response from Groq: {e}"