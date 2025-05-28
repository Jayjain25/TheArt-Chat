import streamlit as st
# No standard Langchain integration for GitHub Copilot

def get_available_models(api_key=None):
    return ["GitHub Copilot (Not directly usable)", "Specify Task/Model"]

def get_llm(api_key, model, params):
    st.error("GitHub Copilot integration via standard LangChain LLM is not available.")
    raise NotImplementedError("GitHub provider does not support get_llm.")

def get_embeddings(api_key=None):
    st.info("GitHub provider does not offer standard text embeddings.")
    return None

def generate_response(api_key, model, messages, params):
    return "Error: GitHub provider interaction is not implemented for standard chat."