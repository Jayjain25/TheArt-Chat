import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.exceptions import OutputParserException
 

# --- Model Listing ---
def get_available_models(api_key):
    if not api_key: return ["Enter API Key"]
    # Basic list, refine as needed or fetch dynamically if API allows
    return [
        "gemini-2.5-pro-preview-03-25",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    ]

# --- LLM Instantiation ---
def get_llm(api_key, model, params):
    if not api_key: raise ValueError("Google API Key missing.")
    try:
        # Map Streamlit params to Google params
        google_params = {
            "temperature": params.get('temperature', 0.7),
            "max_output_tokens": params.get('max_tokens', 1024), # Ensure correct param name
            # "top_p": params.get('top_p'), # Add if used in UI
            # "top_k": params.get('top_k'), # Add if used in UI
        }
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            **google_params,
            convert_system_message_to_human=True # Gemini often prefers this
        )
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Google LLM ({model}): {e}")
        raise

# --- Embeddings ---
def get_embeddings(api_key):
     if not api_key: return None
     try:
         # Use the recommended embedding model
         return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
     except Exception as e:
         st.error(f"Google Embeddings Error: {e}")
         return None

# --- Direct Generation (Optional - if not using ConversationChain always) ---
def generate_response(api_key, model, messages, params):
    """Generates response directly using the LLM instance."""
    try:
        llm = get_llm(api_key, model, params) # Get the LLM instance
        lc_messages = [msg.to_langchain() for msg in messages] # Convert if needed
        response = llm.invoke(lc_messages)
        return response.content
    except OutputParserException as ope:
         st.error(f"Output Parsing Error: {ope}")
         # Sometimes the raw response is still useful
         if hasattr(ope, 'llm_output'): return str(ope.llm_output)
         return f"Error: Output parsing failed. {ope}"
    except Exception as e:
        st.error(f"Google API Error in generate_response ({model}): {e}")
        return f"Error generating response from Google: {e}"