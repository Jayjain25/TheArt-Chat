import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
SITE_URL = "https://openrouter.ai/models" # For model reference

# --- Model Listing ---
def get_available_models(api_key):
    if not api_key: return ["Enter API Key"]
    # It's better for users to refer to the OpenRouter website for the latest models
    # Provide a few examples and the link
    examples = [
        "openai/gpt-4o", "google/gemini-pro-1.5", "anthropic/claude-3-opus",
        "mistralai/mistral-large", "meta-llama/llama-3-70b-instruct",
        # Add more popular ones
    ]
    # You could attempt to fetch dynamically, but it adds complexity/dependency
    # For simplicity, return examples and link
    return examples + [f"See all models at {SITE_URL}"]

# --- LLM Instantiation ---
def get_llm(api_key, model, params):
    if not api_key: raise ValueError("OpenRouter API Key missing.")
    # Check if user selected the info message
    if SITE_URL in model:
         raise ValueError("Please select a specific model ID (e.g., 'google/gemini-flash-1.5'), not the informational link.")
    try:
        # Map generic params to OpenAI compatible ones
        openai_params = {
            "temperature": params.get('temperature', 0.7),
            "max_tokens": params.get('max_tokens', 1024),
            # "model_kwargs": {"top_p": params.get("top_p"), ...} # Use model_kwargs for others
        }
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_API_BASE,
            **openai_params,
            # Pass site URL in headers as recommended by OpenRouter
            default_headers={ "HTTP-Referer": "http://localhost:8501", "X-Title": "TheArt Chat" }
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize OpenRouter LLM ({model}): {e}")
        raise

# --- Embeddings ---
def get_embeddings(api_key):
     if not api_key: return None
     # OpenRouter supports OpenAI embeddings, maybe others? Check their docs.
     # Defaulting to OpenAI's ada-002 via their endpoint
     try:
         return OpenAIEmbeddings(
             model="text-embedding-ada-002", # Or specify another if OR supports it
             openai_api_key=api_key,
             openai_api_base=OPENROUTER_API_BASE,
             default_headers={ "HTTP-Referer": "http://localhost:8501", "X-Title": "TheArt Chat" }
         )
     except Exception as e:
         st.error(f"OpenRouter Embeddings Error: {e}")
         return None

# --- Direct Generation (Optional) ---
def generate_response(api_key, model, messages, params):
    try:
        llm = get_llm(api_key, model, params)
        lc_messages = [msg.to_langchain() for msg in messages]
        response = llm.invoke(lc_messages)
        return response.content
    except Exception as e:
        st.error(f"OpenRouter API Error in generate_response ({model}): {e}")
        return f"Error generating response from OpenRouter: {e}"