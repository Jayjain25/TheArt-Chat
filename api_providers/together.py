import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

TOGETHER_API_BASE = "https://api.together.xyz/v1"

# --- Model Listing ---
def get_available_models(api_key):
    if not api_key: return ["Enter API Key"]
    # Examples from Together AI docs (or fetch dynamically)
    return [
        "mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.1",
        "togethercomputer/llama-2-7b-chat", "Qwen/Qwen1.5-72B-Chat",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        # Add more or link to docs
    ]

# --- LLM Instantiation ---
def get_llm(api_key, model, params):
    if not api_key: raise ValueError("Together API Key missing.")
    try:
        openai_params = {
            "temperature": params.get('temperature', 0.7),
            "max_tokens": params.get('max_tokens', 1024),
        }
        llm = ChatOpenAI(
            model=model,
            api_key=api_key, # Parameter name might be just 'api_key' for newer Langchain OpenAI
            # openai_api_key=api_key, # Or this, check Langchain docs
            base_url=TOGETHER_API_BASE, # Use base_url
            # openai_api_base=TOGETHER_API_BASE, # Or this
            **openai_params
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Together LLM ({model}): {e}")
        raise

# --- Embeddings ---
def get_embeddings(api_key):
     if not api_key: return None
     try:
         # Use a Together-recommended embedding model if known, or a generic one
         # Example: togethercomputer/m2-bert-80M-8k-retrieval (check if still valid)
         # Using OpenAIEmbeddings assumes compatibility
         return OpenAIEmbeddings(
             model="togethercomputer/m2-bert-80M-8k-retrieval", # Verify model name
             api_key=api_key,
             base_url=TOGETHER_API_BASE,
         )
     except Exception as e:
         st.error(f"Together Embeddings Error: {e}")
         return None

# --- Direct Generation (Optional) ---
def generate_response(api_key, model, messages, params):
    try:
        llm = get_llm(api_key, model, params)
        lc_messages = [msg.to_langchain() for msg in messages]
        response = llm.invoke(lc_messages)
        return response.content
    except Exception as e:
        st.error(f"Together API Error ({model}): {e}")
        return f"Error generating response from Together: {e}"