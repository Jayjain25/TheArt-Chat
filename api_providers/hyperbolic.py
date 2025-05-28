import streamlit as st
# Assumes a hypothetical API - no Langchain integration exists

def get_available_models(api_key=None):
    if not api_key: return ["Enter API Key"]
    return ["hyper-model-v1", "hyper-model-pro"] # Fictional

def get_llm(api_key, model, params):
    st.error("Hyperbolic provider integration via standard LangChain LLM is not implemented.")
    raise NotImplementedError("Hyperbolic provider does not support get_llm.")

def get_embeddings(api_key=None):
    st.info("Hyperbolic provider does not offer standard text embeddings.")
    return None

def generate_response(api_key, model, messages, params):
    return f"Error: Hyperbolic provider interaction is not implemented. Placeholder call for {model}."