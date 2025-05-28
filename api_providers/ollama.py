import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import requests

# --- Server Check ---
def check_ollama_server(base_url="http://localhost:11434"):
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3) # Check tags endpoint
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# --- Model Listing ---
def get_available_models(api_key=None, base_url="http://localhost:11434"): # api_key usually ignored
    if not check_ollama_server(base_url):
        return ["Ollama server not running"]
    try:
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()
        models_data = response.json().get("models", [])
        return sorted([model['name'] for model in models_data]) if models_data else ["No models found"]
    except Exception as e:
        st.error(f"Error fetching Ollama models: {e}")
        return ["Error fetching models"]

# --- LLM Instantiation ---
def get_llm(api_key, model, params, base_url="http://localhost:11434"):
    if not check_ollama_server(base_url):
         raise ConnectionError("Ollama server not running or accessible.")
    try:
        # Map generic params to Ollama-specific ones
        ollama_params = {
            "temperature": params.get('temperature'),
            "num_ctx": params.get('num_ctx'), # Context window size
            "top_k": params.get('top_k'),
            "top_p": params.get('top_p'),
            # Add others like 'stop', 'num_predict' (maps roughly to max_tokens)
            "num_predict": params.get('max_tokens'),
        }
        # Filter out None values as ChatOllama might not handle them
        ollama_params = {k: v for k, v in ollama_params.items() if v is not None}

        llm = ChatOllama(
            base_url=base_url,
            model=model,
            **ollama_params
        )
        return llm
    except Exception as e:
        # Check specifically for model not found error if possible
        if "pull" in str(e).lower():
             st.error(f"Model '{model}' not found locally. Pull it using `ollama pull {model}`.", icon="🚨")
             raise ValueError(f"Model '{model}' not found locally.") from e
        st.error(f"Failed to initialize Ollama LLM ({model}): {e}")
        raise

# --- Embeddings ---
def get_embeddings(api_key=None, base_url="http://localhost:11434"):
    if not check_ollama_server(base_url): return None
    try:
        # Try common embedding models first
        available = get_available_models(base_url=base_url)
        common_emb_models = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]
        emb_model = None
        for model_name in common_emb_models:
             if any(model_name in m for m in available):
                  emb_model = model_name
                  break
        if not emb_model and available and "Error" not in available[0] and "not running" not in available[0]:
             st.warning(f"Could not find standard Ollama embedding model (e.g., nomic-embed-text). PDF search might fail. Please run `ollama pull nomic-embed-text`.", icon="⚠️")
             return None # Explicitly return None if no suitable model found
        elif not emb_model:
             st.error("No Ollama embedding models found or server issue.")
             return None

        print(f"Using Ollama embedding model: {emb_model}")
        return OllamaEmbeddings(base_url=base_url, model=emb_model)
    except Exception as e:
         st.error(f"Ollama Embeddings Error: {e}")
         return None

# --- Direct Generation (Optional) ---
def generate_response(api_key, model, messages, params, base_url="http://localhost:11434"):
    try:
        llm = get_llm(api_key, model, params, base_url)
        lc_messages = [msg.to_langchain() for msg in messages]
        response = llm.invoke(lc_messages)
        return response.content
    except Exception as e:
        st.error(f"Ollama API Error in generate_response ({model}): {e}")
        # Provide specific feedback if model needs pulling
        if "pull" in str(e).lower():
            return f"Error: Model '{model}' not found. Pull it using `ollama pull {model}`."
        return f"Error generating response from Ollama: {e}"