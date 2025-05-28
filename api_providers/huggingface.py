import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_core.messages import SystemMessage

# --- Model Listing ---
def get_available_models(api_key):
    if not api_key: return ["Enter API Key / Token"]
    # Provide examples, full dynamic listing is complex via API
    return [
        "mistralai/Mistral-7B-Instruct-v0.1", "google/flan-t5-xxl",
        "meta-llama/Meta-Llama-3-8B-Instruct", # Needs gating
        "HuggingFaceH4/zephyr-7b-beta",
        # User should enter full repo_id from HF Hub
        "Enter full HF Repo ID..."
    ]

# --- LLM Instantiation ---
def get_llm(api_key, model, params):
    if not api_key: raise ValueError("Hugging Face API Token missing.")
    if "Enter full HF Repo ID" in model: raise ValueError("Please enter a specific HF Repo ID.")
    try:
        # Map generic params to HF Endpoint params
        hf_params = {
            "temperature": params.get('temperature', 0.7),
            "max_new_tokens": params.get('max_tokens', 512), # HF often uses max_new_tokens
            "top_p": params.get("top_p"),
            "top_k": params.get("top_k"),
            # Add others like repetition_penalty if needed
        }
        hf_params = {k: v for k, v in hf_params.items() if v is not None}

        llm = HuggingFaceEndpoint(
            repo_id=model, # Expects full repo_id like 'mistralai/Mistral-7B-Instruct-v0.1'
            huggingfacehub_api_token=api_key,
            **hf_params
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize HF Endpoint LLM ({model}): {e}")
        raise

# --- Embeddings ---
def get_embeddings(api_key):
     # Often doesn't require API key unless using private/gated models or specific endpoints
     # Use a standard Sentence Transformer model available on the Hub
     try:
         # BAAI/bge-small-en-v1.5 is a popular choice
         return HuggingFaceEndpointEmbeddings(
             model="BAAI/bge-small-en-v1.5", # Or sentence-transformers/all-MiniLM-L6-v2
             huggingfacehub_api_token=api_key # Pass key if needed by endpoint/model
         )
     except Exception as e:
         st.error(f"HuggingFace Embeddings Error: {e}")
         return None

# --- Direct Generation (Optional) ---
def generate_response(api_key, model, messages, params):
    # HF Endpoint LLM often works better with a formatted prompt string
    # than a list of messages directly. Depends on model/endpoint setup.
    try:
        llm = get_llm(api_key, model, params)
        # --- Simple Prompt Formatting Example ---
        # Needs to match the specific model's expected format (e.g., ChatML, Alpaca)
        prompt = ""
        # Handle system prompt (often needs special tokens)
        system_msg = next((m.content for m in messages if m.role == 'system'), None)
        if system_msg:
             prompt += f"<|system|>\n{system_msg}</s>\n" # Example format, ADJUST PER MODEL

        for msg in messages:
             if msg.role == "user":
                  prompt += f"<|user|>\n{msg.content}</s>\n" # Example format
             elif msg.role == "assistant":
                  prompt += f"<|assistant|>\n{msg.content}</s>\n" # Example format

        # Add final prompt for assistant generation
        prompt += "<|assistant|>\n"

        print(f"DEBUG HF Prompt:\n{prompt}") # Debugging the prompt format

        response = llm.invoke(prompt)
        return response # Response is usually the generated string directly
    except Exception as e:
        st.error(f"HuggingFace API Error ({model}): {e}")
        return f"Error generating response from HuggingFace: {e}"