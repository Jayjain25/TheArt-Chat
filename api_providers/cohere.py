import streamlit as st
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.messages import SystemMessage # Cohere needs system message handling

# --- Model Listing ---
def get_available_models(api_key):
    if not api_key: return ["Enter API Key"]
    # Check Cohere docs for latest models
    return ["command-r-plus", "command-r", "command", "command-light"] # Example list

# --- LLM Instantiation ---
def get_llm(api_key, model, params):
    if not api_key: raise ValueError("Cohere API Key missing.")
    try:
        cohere_params = {
            "temperature": params.get('temperature', 0.7),
            "max_tokens": params.get('max_tokens', 1024),
            # Add other cohere specific params like 'k', 'p' if needed
            # "p": params.get("top_p"),
            # "k": params.get("top_k"),
        }
        llm = ChatCohere(
            model=model,
            cohere_api_key=api_key,
            **cohere_params
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Cohere LLM ({model}): {e}")
        raise

# --- Embeddings ---
def get_embeddings(api_key):
     if not api_key: return None
     try:
         # Check Cohere docs for recommended/latest embedding models
         return CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=api_key)
     except Exception as e:
         st.error(f"Cohere Embeddings Error: {e}")
         return None

# --- Direct Generation (Optional) ---
def generate_response(api_key, model, messages, params):
    try:
        llm = get_llm(api_key, model, params)
        lc_messages = []
        # Cohere specific handling: System message might need to be chat_history or preamble
        system_prompt = ""
        other_messages = []
        for msg in messages:
             lc_msg = msg.to_langchain()
             if isinstance(lc_msg, SystemMessage):
                  # Handle system prompt based on ChatCohere's expectation (check docs)
                  # Option 1: Use preamble (if supported directly in constructor/params)
                  # Option 2: Pass it as the first 'user' message if preamble not simple
                  # Option 3: Ignore if LLM doesn't use it well.
                  system_prompt = lc_msg.content # Store it for now
             else:
                  other_messages.append(lc_msg)

        # Example using preamble if ChatCohere supports it (check constructor args)
        # llm = ChatCohere(..., preamble=system_prompt)
        # response = llm.invoke(other_messages)

        # Or handle manually if needed
        # This might depend on exact ChatCohere implementation details
        response = llm.invoke(other_messages) # Simplest approach first

        return response.content
    except Exception as e:
        st.error(f"Cohere API Error ({model}): {e}")
        return f"Error generating response from Cohere: {e}"