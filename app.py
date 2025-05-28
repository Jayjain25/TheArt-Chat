# --- Start of app.py ---
import streamlit as st
import importlib
import os
import datetime
import time
import json
import tiktoken # Added for token counting
from pathlib import Path
from api_providers.nvidia_nim import NvidiaProvider

# Project specific imports
try:
    from utils.helpers import Message, get_timestamp_id, export_chat_to_txt, export_chat_to_csv, calculate_analytics
    from utils.pdf_processor import process_pdfs, get_relevant_context
    from utils.storage import (
        save_chat, load_chats, delete_chat_file,
        save_api_key, get_api_key, delete_api_key,
        load_saved_models, save_saved_models,
        save_tavily_api_key, get_tavily_api_key, delete_tavily_api_key,
        # --- Add functions for prompt templates ---
        load_prompt_templates, save_prompt_templates
    )
except ImportError as e:
    st.error(f"Import Error: {e}. Check 'utils' directory and ensure prompt template functions exist in storage.py."); st.stop()

# LangChain imports for core logic
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# LangChain imports for Agent/Tools
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub

# --- Configuration & Setup ---
APP_TITLE = "TheArt Chat"
PROVIDERS_DIR = "api_providers"
if not os.path.isdir(PROVIDERS_DIR): st.error(f"Directory '{PROVIDERS_DIR}' not found."); st.stop()
AVAILABLE_PROVIDERS = sorted([f[:-3] for f in os.listdir(PROVIDERS_DIR) if f.endswith(".py") and f != "__init__.py"])
if not AVAILABLE_PROVIDERS: st.error(f"No providers found in '{PROVIDERS_DIR}'."); st.stop()
provider_modules = {}
for provider_name in list(AVAILABLE_PROVIDERS):
    try: provider_modules[provider_name] = importlib.import_module(f"{PROVIDERS_DIR}.{provider_name}")
    except ImportError as e: st.error(f"Import failed for {provider_name}: {e}"); AVAILABLE_PROVIDERS.remove(provider_name)

PROVIDERS = {
    "cohere": "Cohere",
    "github": "GitHub",
    "google": "Google",
    "groqcloud": "GroqCloud",
    "huggingface": "Hugging Face",
    "hyperbolic": "Hyperbolic",
    "nvidia_nim": "NVIDIA NIM",
    "ollama": "Ollama",
    "openrouter": "OpenRouter",
    "tavily": "Tavily (Web Search)",
    "together": "Together"
}

# --- Constants ---
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MEMORY_TYPE = "summary_buffer"
DEFAULT_MEMORY_CONFIG = {"max_token_limit": 600}
AVAILABLE_MEMORY_TYPES = {
    "summary_buffer": "Summary Buffer (Summarizes older messages)",
    "buffer_window": "Buffer Window (Keeps last K messages)"
}
DEFAULT_TOKENIZER = "cl100k_base" # Default for many OpenAI models

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(f"🎨 {APP_TITLE}")


# --- Tokenizer Helper ---
@st.cache_resource
def get_tokenizer(encoding_name=DEFAULT_TOKENIZER):
    """Gets a TikToken tokenizer instance."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        print(f"Warning: Failed to get tokenizer '{encoding_name}': {e}. Falling back to primitive count.")
        return None

def count_tokens(text, tokenizer):
    """Counts tokens in a string using the provided tokenizer."""
    if tokenizer and isinstance(text, str): # Added check for string type
        return len(tokenizer.encode(text))
    elif isinstance(text, str):
        # Fallback: very rough estimate
        return len(text.split())
    else:
        return 0 # Cannot count tokens if not a string


# --- Initialize Session State ---
def initialize_chat_data(chat_data):
    """Ensure chat data fields have default values."""
    chat_data.setdefault("system_prompt", DEFAULT_SYSTEM_PROMPT)
    chat_data.setdefault("memory_type", DEFAULT_MEMORY_TYPE)
    chat_data.setdefault("memory_config", DEFAULT_MEMORY_CONFIG.copy())
    return chat_data

def update_token_count_indicator(): # Defined earlier now, needed in load_initial_state
    """Estimates and updates the token count in session state for the current chat."""
    chat_id = st.session_state.current_chat_id
    if chat_id and chat_id in st.session_state.chats:
        messages = st.session_state.chats[chat_id].get("messages", [])
        tokenizer = get_tokenizer() # Use cached tokenizer
        total_tokens = 0
        for msg in messages:
             # Ensure content exists and is a string before counting
             content = getattr(msg, 'content', None)
             if isinstance(content, str):
                 total_tokens += count_tokens(f"{msg.role}: {content}\n", tokenizer)
        st.session_state.current_estimated_tokens = total_tokens
    else:
        st.session_state.current_estimated_tokens = 0

def load_initial_state():
    if "app_loaded" not in st.session_state:
        print("--- Initializing App State ---")
        st.session_state.chats = load_chats()
        for chat_id in st.session_state.chats:
            st.session_state.chats[chat_id] = initialize_chat_data(st.session_state.chats[chat_id])

        st.session_state.saved_models = load_saved_models()
        st.session_state.api_keys = { p: get_api_key(p) or "" for p in AVAILABLE_PROVIDERS }
        st.session_state.tavily_api_key = get_tavily_api_key() or ""

        # --- Initialize Prompt Templates ---
        st.session_state.prompt_templates = load_prompt_templates() # Load templates
        # --- End Initialization ---

        st.session_state.current_chat_id = max(st.session_state.chats.keys(), default=None) if st.session_state.chats else None
        initial_provider = AVAILABLE_PROVIDERS[0] if AVAILABLE_PROVIDERS else None

        if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
            chat_data = st.session_state.chats[st.session_state.current_chat_id]
            loaded_provider = chat_data.get("provider")
            if loaded_provider and loaded_provider in AVAILABLE_PROVIDERS: initial_provider = loaded_provider
            st.session_state.selected_model = chat_data.get("model")
            st.session_state.model_params = chat_data.get("params", {"temperature": 0.7, "max_tokens": 1024})

        st.session_state.setdefault('selected_provider', initial_provider)
        st.session_state.setdefault('available_models', [])
        st.session_state.setdefault('model_params', {"temperature": 0.7, "max_tokens": 1024})
        st.session_state.setdefault('uploaded_files', [])
        st.session_state.setdefault('vector_store', None)
        st.session_state.setdefault('pdf_questioning_mode', False)
        st.session_state.setdefault('web_search_enabled', False)
        st.session_state.setdefault('embedding_function', None)
        st.session_state.setdefault('embedding_checked_provider', None)
        st.session_state.setdefault('_last_provider_for_models', None)
        st.session_state.setdefault('manual_context_injection', "") # For context injection
        st.session_state.setdefault('current_estimated_tokens', 0) # For token indicator

        print(f"Initial State: Provider={st.session_state.selected_provider}, ChatID={st.session_state.current_chat_id}")
        st.session_state.app_loaded = True
        update_token_count_indicator() # Update token count after initial load

load_initial_state()

# --- Helper Functions ---
def get_provider_module(provider_name): return provider_modules.get(provider_name)

def update_models_list():
    provider_name = st.session_state.selected_provider
    if not provider_name: return

    api_key = st.session_state.api_keys.get(provider_name, "")
    module = get_provider_module(provider_name)
    original_model = st.session_state.get('selected_model')
    st.session_state.available_models = []
    st.session_state.selected_model = None

    if module and hasattr(module, "get_available_models"):
        models = []; show_spinner = api_key or provider_name not in ['ollama']
        with st.spinner(f"Fetching models...") if show_spinner else st.empty():
            try: models = module.get_available_models(api_key)
            except Exception as e: models = ["Error fetching models"]; print(f"Fetch models error: {e}")
        saved = st.session_state.saved_models.get(provider_name, [])
        models = [m for m in models if isinstance(m, str)]; saved = [s for s in saved if isinstance(s, str)]
        combined = sorted(list(set(models + saved)))
        error_indicators = ["Error", "server not running", "Enter API Key", "No models found"]
        combined = [m for m in combined if not any(indicator in m for indicator in error_indicators)]
        st.session_state.available_models = combined
        if original_model in combined: st.session_state.selected_model = original_model
        elif combined: st.session_state.selected_model = combined[0]
        print(f"Models updated for {provider_name}. Selected: {st.session_state.selected_model}")
    else:
        st.session_state.available_models = ["Provider module error"]
        print(f"Provider module error or missing function for {provider_name}.")
    st.session_state._last_provider_for_models = provider_name

def create_new_chat():
    chat_id = get_timestamp_id(); st.session_state.current_chat_id = chat_id
    chat_data = {
        "name": f"Chat_{chat_id[-6:]}",
        "messages": [Message("system", DEFAULT_SYSTEM_PROMPT)],
        "provider": st.session_state.selected_provider,
        "model": st.session_state.selected_model,
        "params": st.session_state.model_params.copy(),
        "memory": None,
        "vector_store": None,
        "needs_reprocessing": False,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "memory_type": DEFAULT_MEMORY_TYPE,
        "memory_config": DEFAULT_MEMORY_CONFIG.copy(),
    }
    st.session_state.chats[chat_id] = chat_data
    save_chat(chat_id, chat_data)
    st.session_state.vector_store = None; st.session_state.uploaded_files = []; st.session_state.pdf_questioning_mode = False
    st.session_state.manual_context_injection = "" # Reset context injection on new chat
    st.session_state.current_estimated_tokens = 0 # Reset token count
    update_token_count_indicator() # Update indicator for the new empty chat
    st.info(f"Started: {chat_data['name']}")

def switch_chat(chat_id):
    print(f"--- switch_chat called for ID: {chat_id} ---")
    if chat_id in st.session_state.chats:
        st.session_state.current_chat_id = chat_id
        chat_data = st.session_state.chats[chat_id]
        chat_data = initialize_chat_data(chat_data) # Ensure fields exist
        st.session_state.chats[chat_id] = chat_data

        loaded_provider = chat_data.get("provider")
        if loaded_provider and loaded_provider in AVAILABLE_PROVIDERS:
            st.session_state.selected_provider = loaded_provider
        else:
            fallback_provider = AVAILABLE_PROVIDERS[0] if AVAILABLE_PROVIDERS else None
            st.session_state.selected_provider = fallback_provider

        st.session_state.selected_model = chat_data.get("model")
        st.session_state.model_params = chat_data.get("params", {"temperature": 0.7, "max_tokens": 1024})

        st.session_state.embedding_function = None; st.session_state.embedding_checked_provider = None; st.session_state.vector_store = None; st.session_state.uploaded_files = []; st.session_state.pdf_questioning_mode = False; st.session_state.web_search_enabled = False
        st.session_state.manual_context_injection = "" # Reset context injection on switch
        update_token_count_indicator() # Update token count for the switched chat

        update_models_list()
        print(f"--- switch_chat finished, calling rerun ---")
        st.rerun()
    else: st.error("Chat ID not found.")

# --- Handle User Input (Handles Agent vs Chain) ---
def handle_user_input(prompt: str, is_regenerate: bool = False):
    print(f"--- handle_user_input (Regen: {is_regenerate}) ---"); chat_id = st.session_state.current_chat_id
    if not chat_id or chat_id not in st.session_state.chats: st.error("No active chat."); return
    current_chat = initialize_chat_data(st.session_state.chats[chat_id])

    # --- Handle Manual Context Injection ---
    injected_context = st.session_state.get("manual_context_injection", "")
    final_user_prompt = prompt # Store original prompt before potential modification
    if injected_context:
        print("Injecting manual context for this turn.")
        final_user_prompt = f"Consider this context:\n---\n{injected_context}\n---\n\nUser Question: {prompt}"
        st.session_state.manual_context_injection = "" # Clear after use
        st.toast("Manual context injected for this message.", icon="💉")
    # --- End Context Injection ---

    if not is_regenerate:
        # Store the potentially modified prompt (with context) as the user message
        current_chat["messages"].append(Message("user", final_user_prompt))

    provider_name = current_chat.get("provider"); model_name = current_chat.get("model")
    api_key = st.session_state.api_keys.get(provider_name, ""); params = current_chat.get("params")
    module = get_provider_module(provider_name)
    if not all([provider_name, model_name, params, module]) or model_name in ["Enter API Key", "No models found", "Provider module error", "Error fetching models"]: st.error("Model/Provider Configuration error. Check sidebar."); st.rerun(); return
    provider_needs_key = provider_name not in ['ollama'];
    if provider_needs_key and not api_key: st.error(f"API Key needed for {provider_name}. Please enter it in the sidebar."); st.rerun(); return
    if not hasattr(module, "get_llm"): st.error(f"Provider module {provider_name} missing 'get_llm' function."); st.rerun(); return

    llm_instance = None
    try: llm_instance = module.get_llm(api_key=api_key, model=model_name, params=params)
    except Exception as e: st.error(f"LLM Initialization Error for {model_name}: {e}"); st.rerun(); return
    if not llm_instance: st.error(f"Could not initialize LLM instance {model_name}."); st.rerun(); return

    memory_type = current_chat["memory_type"]
    memory_config = current_chat["memory_config"]
    system_prompt = current_chat["system_prompt"]

    if current_chat["messages"] and current_chat["messages"][0].role == "system":
        if current_chat["messages"][0].content != system_prompt:
            current_chat["messages"][0] = Message("system", system_prompt)
            current_chat["memory"] = None
            print("System prompt changed, memory will be reset.")
    elif not current_chat["messages"] or current_chat["messages"][0].role != "system":
         current_chat["messages"].insert(0, Message("system", system_prompt))
         current_chat["memory"] = None
         print("System prompt added, memory will be reset.")

    if is_regenerate or current_chat.get("memory") is None:
        print(f"Init/Reset memory (Type: {memory_type}, Config: {memory_config})");
        try:
            if memory_type == "summary_buffer":
                mem_instance = ConversationSummaryBufferMemory(
                    llm=llm_instance, max_token_limit=memory_config.get("max_token_limit", 600),
                    memory_key="history", input_key="input", return_messages=True,
                    system_message=SystemMessage(content=system_prompt) if system_prompt else None
                )
            elif memory_type == "buffer_window":
                mem_instance = ConversationBufferWindowMemory(
                    k=memory_config.get("k", 5), memory_key="history", input_key="input", return_messages=True,
                )
            else:
                mem_instance = ConversationSummaryBufferMemory(llm=llm_instance, max_token_limit=600, memory_key="history", input_key="input", return_messages=True)

            current_chat["memory"] = mem_instance
            messages_for_memory = current_chat["messages"][:-1] if not is_regenerate else current_chat["messages"]
            inputs = []; outputs = []
            for i in range(1, len(messages_for_memory), 2):
                if i+1 < len(messages_for_memory) and messages_for_memory[i].role == "user" and messages_for_memory[i+1].role == "assistant":
                    inputs.append({"input": messages_for_memory[i].content})
                    outputs.append({"output": messages_for_memory[i+1].content})
            if inputs and outputs:
                current_chat["memory"].save_context(inputs=inputs, outputs=outputs)
                print(f"Loaded {len(inputs)} pairs into fresh memory.")
        except Exception as e:
            st.error(f"Failed to initialize memory: {e}"); st.rerun(); return

    chat_memory = current_chat["memory"]

    # Pass the potentially modified prompt (with context) to the LLM/Agent
    prompt_for_llm = final_user_prompt

    active_vector_store = st.session_state.get("vector_store"); use_pdf_context = False
    if st.session_state.pdf_questioning_mode and active_vector_store:
        print("PDF Mode Active"); st.session_state.web_search_enabled = False
        # Use original prompt for context search, but modified prompt for final LLM call
        with st.spinner("Searching PDF context..."): context = get_relevant_context(prompt, active_vector_store)
        if isinstance(context, str) and "Error" not in context and "initialized" not in context and context.strip():
            # Prepend PDF context to the prompt_for_llm (which might already have manual context)
            prompt_for_llm = f"Based on the following context:\n---\n{context}\n---\nPlease answer this question: {prompt_for_llm}"
            print("Using retrieved PDF context."); st.sidebar.caption("Using PDF context."); use_pdf_context = True
        else:
             print("PDF context retrieval failed or empty, answering generally."); st.warning("Could not get relevant PDF context. Answering generally.")
    elif st.session_state.pdf_questioning_mode:
        st.warning("PDF Mode is ON, but no PDF data is processed or available."); st.session_state.pdf_questioning_mode = False

    response_content = None; error_occurred = False; execution_mode = "Chain"
    agent_prompt_object = None

    if not use_pdf_context and st.session_state.web_search_enabled:
         execution_mode = "Agent"; print("Web Search Active"); tavily_api_key = st.session_state.tavily_api_key
         if not tavily_api_key: st.error("Tavily API key needed for web search."); st.rerun(); return
         with st.spinner(f"Running Agent with Web Search..."):
             try:
                 tavily_tool = TavilySearchResults(api_key=tavily_api_key, max_results=3); tools = [tavily_tool]
                 try:
                     agent_prompt_object = hub.pull("hwchase17/openai-tools-agent")
                     if system_prompt and isinstance(agent_prompt_object.messages[0], SystemMessage):
                         agent_prompt_object.messages[0].content = system_prompt; print("Injected system prompt into agent prompt.")
                     elif system_prompt:
                         agent_prompt_object.messages.insert(0, SystemMessage(content=system_prompt)); print("Prepended system prompt to agent prompt.")
                 except Exception as e: st.error(f"Failed to pull agent prompt: {e}"); st.rerun(); return
                 try: agent = create_openai_tools_agent(llm_instance, tools, agent_prompt_object)
                 except Exception as e: st.error(f"Failed to create agent: {e}"); st.rerun(); return
                 agent_executor = AgentExecutor(agent=agent, tools=tools, memory=chat_memory, verbose=True, handle_parsing_errors="Check your output and make sure it conforms!", max_iterations=5)
                 print(f"Invoking Agent for: {prompt_for_llm[:100]}...") # Use prompt_for_llm
                 response = agent_executor.invoke({"input": prompt_for_llm}) # Use prompt_for_llm
                 response_content = response.get('output')
                 if response_content is None: raise ValueError("Agent output key 'output' not found in response.")
                 print(f"Agent Response Content: {str(response_content)[:100]}...")
             except Exception as e: print(f"Agent Execution Error: {e}"); st.error(f"Agent Error: {e}"); response_content = f"Sorry, I encountered an error while searching the web: {e}"; error_occurred = True
    else:
        print(f"Normal Mode ({'PDF Context' if use_pdf_context else 'No PDF/Web'}): Using ConversationChain.")
        with st.spinner(f"Generating response..."):
             try:
                 conversation = ConversationChain(llm=llm_instance, memory=chat_memory, verbose=False)
                 print(f"Running chain for: {prompt_for_llm[:100]}...") # Use prompt_for_llm
                 response = conversation.invoke({"input": prompt_for_llm}) # Use prompt_for_llm
                 response_content = response.get('response')
                 if response_content is None: raise ValueError("Chain output key 'response' not found in response.")
                 print(f"Chain Response Content: {str(response_content)[:100]}...")
             except Exception as e: print(f"Chain Error: {e}"); st.error(f"Chain Error: {e}"); response_content = f"Sorry, I encountered an error generating the response: {e}"; error_occurred = True

    print(f"Updating chat display after {execution_mode}.")
    if not current_chat["messages"] or current_chat["messages"][-1].role == "user":
        current_chat["messages"].append(Message("assistant", response_content))
        print("Appended new assistant message.")
    elif current_chat["messages"][-1].role == "assistant" and is_regenerate:
        current_chat["messages"][-1] = Message("assistant", response_content)
        print("Overwrote last assistant message for regeneration.")
    else:
        print(f"Warning: Unexpected message sequence before adding assistant response. Last msg role: {current_chat['messages'][-1].role}")
        current_chat["messages"].append(Message("assistant", response_content))

    current_chat["last_provider"] = provider_name; current_chat["last_model"] = model_name
    save_chat(chat_id, current_chat); print("Chat state saved.")
    update_token_count_indicator() # Update token count after adding messages
    print(f"--- handle_user_input finished, calling st.rerun() ---")
    st.rerun()


# --- Regenerate Response ---
def regenerate_response():
    print("--- regenerate_response called ---"); chat_id = st.session_state.current_chat_id
    if not (chat_id and chat_id in st.session_state.chats): print("Regen failed: No active chat."); return
    current_chat = st.session_state.chats[chat_id]; messages = current_chat["messages"]; last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == 'user': last_user_idx = i; break
    if last_user_idx != -1:
        # Get the user prompt as it was originally stored (might include injected context)
        user_prompt_to_regenerate = messages[last_user_idx].content
        print(f"Regenerating from last user prompt content: {user_prompt_to_regenerate[:50]}...")
        current_chat["messages"] = messages[:last_user_idx + 1] # Trim messages
        print(f"Display history trimmed to index {last_user_idx}.")
        current_chat["memory"] = None; print("Chat memory reset for regeneration.")
        save_chat(chat_id, current_chat); print("Trimmed state saved before regeneration.")
        # Pass the original user prompt content to handle_user_input for regeneration
        # handle_user_input will NOT inject context again because manual_context_injection is cleared after use
        handle_user_input(user_prompt_to_regenerate, is_regenerate=True)
    else:
        print("Regen failed: No user message found in history."); st.warning("Cannot regenerate without a previous user message.")


# --- Sidebar UI ---
with st.sidebar:
    st.header("Configuration")

    # --- Provider Selection ---
    # (Code omitted for brevity - same as before)
    current_provider_in_state = st.session_state.get('selected_provider')
    if current_provider_in_state not in AVAILABLE_PROVIDERS: current_provider_in_state = AVAILABLE_PROVIDERS[0] if AVAILABLE_PROVIDERS else None
    provider_index = AVAILABLE_PROVIDERS.index(current_provider_in_state) if current_provider_in_state in AVAILABLE_PROVIDERS else 0
    selected_provider_widget_value = st.selectbox(
        "API Provider", AVAILABLE_PROVIDERS, key="provider_select", index=provider_index
    )
    if selected_provider_widget_value != current_provider_in_state:
        st.session_state.selected_provider = selected_provider_widget_value
        st.session_state.api_keys[st.session_state.selected_provider] = get_api_key(st.session_state.selected_provider) or ""
        st.session_state.embedding_function = None; st.session_state.embedding_checked_provider = None
        st.session_state.selected_model = None
        update_models_list()
        if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
            st.session_state.chats[st.session_state.current_chat_id]['provider'] = selected_provider_widget_value
            save_chat(st.session_state.current_chat_id, st.session_state.chats[st.session_state.current_chat_id])
        st.rerun()

    # --- API Key Input ---
    provider_name = st.session_state.selected_provider
    if provider_name:
        # (Code omitted for brevity - same as before)
        loaded_key = st.session_state.api_keys.get(provider_name, ""); is_key_stored = bool(loaded_key)
        st.caption(f"{'✅ Key Stored' if is_key_stored else 'ℹ️ Key Not Stored'} for {provider_name}")
        api_key_input = st.text_input(f"Enter/Update Key", type="password", key=f"api_input_{provider_name}", placeholder="Enter key..." if not is_key_stored else "Enter key to update...")
        cs1, cs2 = st.columns(2)
        if cs1.button(f"Save Key", key=f"save_key_{provider_name}", disabled=not api_key_input, use_container_width=True): st.session_state.api_keys[provider_name] = api_key_input; save_api_key(provider_name, api_key_input); st.success(f"Key saved.", icon="🔐"); st.session_state.embedding_function = None; st.session_state.embedding_checked_provider = None; update_models_list(); time.sleep(0.5); st.rerun()
        if cs2.button(f"Clear Key", key=f"clear_key_{provider_name}", disabled=not is_key_stored, use_container_width=True): delete_api_key(provider_name); st.session_state.api_keys[provider_name] = ""; st.toast(f"Key cleared.", icon="🗑️"); st.session_state.embedding_function = None; st.session_state.embedding_checked_provider = None; update_models_list(); st.rerun()
    else: st.warning("No provider selected.")

    # --- Model Selection ---
    st.subheader("Model")
    if provider_name:
        # (Code omitted for brevity - same as before)
        if st.session_state._last_provider_for_models != provider_name: update_models_list()
        current_model = st.session_state.get('selected_model'); available_models = st.session_state.get('available_models', [])
        current_idx = available_models.index(current_model) if current_model in available_models else 0
        model_disabled = not bool(available_models) or ("Error" in available_models[0] if available_models else False)
        selected_model_widget_value = st.selectbox("Select Model", options=available_models, key="model_select", index=current_idx, disabled=model_disabled)
        if selected_model_widget_value != current_model:
             st.session_state.selected_model = selected_model_widget_value
             if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
                 st.session_state.chats[st.session_state.current_chat_id]['model'] = selected_model_widget_value
                 save_chat(st.session_state.current_chat_id, st.session_state.chats[st.session_state.current_chat_id])
             print(f"Model selection changed to: {selected_model_widget_value}")
    else: 
        st.caption("Select provider.")

    # Manual Model Management
    if provider_name:
        with st.expander("Manual Model Management"):
            # Initialize saved models in session state if not exists
            if 'saved_models' not in st.session_state:
                st.session_state.saved_models = {}
            if provider_name not in st.session_state.saved_models:
                st.session_state.saved_models[provider_name] = []
            
            # Add new model
            new_model = st.text_input("Enter new model name", key="new_model_input")
            if st.button("Add Model"):
                if new_model and new_model not in st.session_state.saved_models[provider_name]:
                    st.session_state.saved_models[provider_name].append(new_model)
                    st.success(f"Model '{new_model}' added successfully!")
                    st.experimental_rerun()
            
            # List and manage existing custom models
            if st.session_state.saved_models[provider_name]:
                st.subheader("Manage Custom Models")
                for model in st.session_state.saved_models[provider_name]:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.text(model)
                    with col2:
                        if st.button("Edit", key=f"edit_{model}"):
                            st.session_state.editing_model = model
                    with col3:
                        if st.button("Delete", key=f"delete_{model}"):
                            st.session_state.saved_models[provider_name].remove(model)
                            st.success(f"Model '{model}' deleted successfully!")
                            st.experimental_rerun()
                
                # Edit model name
                if 'editing_model' in st.session_state and st.session_state.editing_model:
                    edited_name = st.text_input("Edit model name", 
                                             value=st.session_state.editing_model,
                                             key="edit_model_input")
                    if st.button("Save Changes"):
                        idx = st.session_state.saved_models[provider_name].index(st.session_state.editing_model)
                        st.session_state.saved_models[provider_name][idx] = edited_name
                        st.session_state.editing_model = None
                        st.success(f"Model name updated successfully!")
                        st.experimental_rerun()

    # --- Advanced Model Parameters ---
    with st.expander("Advanced Model Parameters"):
        # (Code omitted for brevity - same as before)
        current_params = st.session_state.model_params # Still use session state as intermediate
        temp_val = float(current_params.get("temperature", 0.7))
        tokens_val = int(current_params.get("max_tokens", 1024))
        new_temp = st.slider("Temperature", 0.0, 2.0, temp_val, 0.1, key="p_temp")
        new_tokens = st.number_input("Max Tokens", 50, 16384, tokens_val, 64, key="p_max")
        params_changed = False
        if new_temp != temp_val:
            st.session_state.model_params["temperature"] = new_temp
            params_changed = True
        if new_tokens != tokens_val:
            st.session_state.model_params["max_tokens"] = new_tokens
            params_changed = True
        if params_changed:
            print(f"Params changed to: Temp={new_temp}, Tokens={new_tokens}")
            if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
                st.session_state.chats[st.session_state.current_chat_id]['params'] = st.session_state.model_params.copy()
                save_chat(st.session_state.current_chat_id, st.session_state.chats[st.session_state.current_chat_id])

    # --- Token Count Indicator ---
    st.caption(f"Est. Tokens in History: {st.session_state.get('current_estimated_tokens', 0)}")
    st.caption(f"_Using '{DEFAULT_TOKENIZER}' for estimation._")

    st.divider()

    # --- System Prompt & Personas ---
    st.subheader("System Prompt")
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
        # (Code omitted for brevity - same as before)
        current_chat_data_sys = initialize_chat_data(st.session_state.chats[st.session_state.current_chat_id])
        chat_id_sys = st.session_state.current_chat_id
        personas = {
            "Helpful Assistant": "You are a helpful assistant.",
            "Code Generator": "You are a code generation assistant. Respond only with code in markdown blocks.",
            "Creative Writer": "You are a creative writing assistant. Focus on storytelling and evocative language.",
            "Sarcastic Bot": "You are a sarcastic assistant. Your responses should be witty and slightly mocking.",
            "Custom": current_chat_data_sys['system_prompt']
        }
        selected_persona_name = st.selectbox(
            "Select Persona (or Edit Custom)", options=list(personas.keys()), key=f"persona_select_{chat_id_sys}",
            index=list(personas.keys()).index("Custom")
        )
        current_system_prompt = personas.get(selected_persona_name, current_chat_data_sys['system_prompt'])
        new_system_prompt = st.text_area(
            "Edit System Prompt:", value=current_system_prompt, key=f"system_prompt_input_{chat_id_sys}", height=150
        )
        if st.button("Apply System Prompt", key=f"apply_sys_prompt_{chat_id_sys}"):
            if new_system_prompt != current_chat_data_sys['system_prompt']:
                st.session_state.chats[chat_id_sys]['system_prompt'] = new_system_prompt
                st.session_state.chats[chat_id_sys]['memory'] = None # Force memory reset
                save_chat(chat_id_sys, st.session_state.chats[chat_id_sys])
                st.success("System prompt updated. Memory will reset on next message.")
                st.rerun()
            else: st.info("System prompt unchanged.")
    else: st.caption("Select a chat to manage its system prompt.")

    st.divider()

    # --- Prompt Templates ---
    st.subheader("Prompt Templates")
    # Ensure prompt_templates exists in session state (should be handled by load_initial_state)
    templates = st.session_state.get("prompt_templates", {})
    template_names = ["<Select a template>"] + sorted(list(templates.keys()))

    selected_template_name = st.selectbox("Load Template", options=template_names, key="template_select")

    if selected_template_name != "<Select a template>":
        template_content = templates.get(selected_template_name, "") # Use .get for safety
        st.text_area("Template Content (Copy text below)", value=template_content, height=150, key="template_display_area", disabled=True)
        st.caption("Manually copy the text above to use it in the chat input.")

    with st.expander("Manage Templates"):
        # Use unique keys for the input fields inside the expander
        new_template_name = st.text_input("New Template Name", key="new_template_name_input")
        new_template_content = st.text_area("New Template Content", key="new_template_content_input", height=100)
        if st.button("Save New Template", key="save_template_btn", disabled=not new_template_name or not new_template_content):
            if new_template_name in templates:
                 st.warning(f"Template '{new_template_name}' already exists. Overwrite?")
                 # Basic overwrite without confirmation for now
            st.session_state.prompt_templates[new_template_name] = new_template_content
            save_prompt_templates(st.session_state.prompt_templates)
            st.success(f"Template '{new_template_name}' saved.")
            # No easy way to clear inputs after saving without callbacks, leave as is. User can manually clear.
            st.rerun() # Refresh the selectbox

        st.markdown("---")
        if templates:
            st.write("Delete Templates:")
            num_cols = 2
            cols = st.columns(num_cols)
            template_list = sorted(list(templates.keys()))
            for i, name in enumerate(template_list):
                col_index = i % num_cols
                if cols[col_index].button(f"🗑️ {name}", key=f"delete_template_{name}", use_container_width=True):
                    del st.session_state.prompt_templates[name]
                    save_prompt_templates(st.session_state.prompt_templates)
                    st.success(f"Template '{name}' deleted.")
                    st.rerun() # Refresh list
        else:
            st.caption("No templates saved yet.")

    st.divider()

    # --- Manual Context Injection ---
    st.subheader("Manual Context Injection")
    # Use a different key to avoid conflicts if 'manual_context_injection' is used elsewhere
    context_area_content = st.text_area(
        "Context to inject for the *next* message:",
        key="manual_context_area_input", # Use a unique key for the widget
        value=st.session_state.get("manual_context_injection", ""), # Display staged context
        height=100,
        help="This text will be prepended to your next chat message only."
        )
    if st.button("Stage Context for Next Turn", key="inject_context_btn"):
        if context_area_content:
            st.session_state.manual_context_injection = context_area_content # Store the content from the area
            st.success("Context staged for injection on next message send.")
            # The text area will retain the value until the next message is sent
        else:
            st.warning("Context area is empty.")
            st.session_state.manual_context_injection = "" # Clear staged context if area is empty

    st.divider()

    # --- Memory Management ---
    st.subheader("Memory Settings")
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
        # (Code omitted for brevity - same as before)
        current_chat_data_mem = initialize_chat_data(st.session_state.chats[st.session_state.current_chat_id])
        chat_id_mem = st.session_state.current_chat_id
        current_mem_type = current_chat_data_mem['memory_type']
        mem_type_options = list(AVAILABLE_MEMORY_TYPES.keys())
        mem_type_display = [AVAILABLE_MEMORY_TYPES[k] for k in mem_type_options]
        current_mem_idx = mem_type_options.index(current_mem_type) if current_mem_type in mem_type_options else 0
        selected_mem_type = st.selectbox(
            "Memory Type", options=mem_type_options, index=current_mem_idx,
            format_func=lambda k: AVAILABLE_MEMORY_TYPES[k], key=f"mem_type_{chat_id_mem}"
        )
        current_mem_config = current_chat_data_mem['memory_config'].copy(); new_mem_config = current_mem_config.copy()
        if selected_mem_type == "summary_buffer":
            max_tokens = current_mem_config.get("max_token_limit", 600)
            new_max_tokens = st.number_input( "Max Token Limit (for Summary)", min_value=50, max_value=4000, value=max_tokens, step=50, key=f"mem_config_tokens_{chat_id_mem}")
            new_mem_config["max_token_limit"] = new_max_tokens; new_mem_config.pop("k", None)
        elif selected_mem_type == "buffer_window":
            k_value = current_mem_config.get("k", 5)
            new_k_value = st.number_input("Window Size (K messages)", min_value=1, max_value=50, value=k_value, step=1, key=f"mem_config_k_{chat_id_mem}")
            new_mem_config["k"] = new_k_value; new_mem_config.pop("max_token_limit", None)
        if st.button("Apply Memory Settings", key=f"apply_mem_settings_{chat_id_mem}"):
            if selected_mem_type != current_mem_type or new_mem_config != current_mem_config:
                st.session_state.chats[chat_id_mem]['memory_type'] = selected_mem_type
                st.session_state.chats[chat_id_mem]['memory_config'] = new_mem_config
                st.session_state.chats[chat_id_mem]['memory'] = None # Force reset
                save_chat(chat_id_mem, st.session_state.chats[chat_id_mem])
                st.success("Memory settings updated. Memory will reset on next message."); st.rerun()
            else: st.info("Memory settings unchanged.")
        mem_c1, mem_c2 = st.columns(2)
        with mem_c1:
            if st.button("Clear Chat Memory", key=f"clear_mem_{chat_id_mem}", use_container_width=True):
                st.session_state.chats[chat_id_mem]['memory'] = None; save_chat(chat_id_mem, st.session_state.chats[chat_id_mem]); st.toast("Chat memory cleared for next interaction.")
        with mem_c2:
             with st.expander("View Memory State", expanded=False):
                 # (Code omitted for brevity - same as before)
                 pass
    else: st.caption("Select a chat to manage its memory.")

    st.divider()

    # --- Tavily Web Search Section ---
    st.subheader("Web Search")
    # (Code omitted for brevity - same as before)
    tavily_key_stored = bool(st.session_state.tavily_api_key)
    st.caption(f"{'✅ Key Stored' if tavily_key_stored else 'ℹ️ Key Not Stored'}")
    tavily_key_input = st.text_input("Tavily API Key", type="password", key="tavily_input", placeholder="Enter key..." if not tavily_key_stored else "Enter key to update...")
    tav_c1, tav_c2 = st.columns(2)
    if tav_c1.button("Save Tavily Key", key="save_tav_btn", disabled=not tavily_key_input, use_container_width=True): st.session_state.tavily_api_key = tavily_key_input; save_tavily_api_key(tavily_key_input); st.success("Tavily key saved.", icon="🔑"); time.sleep(0.5); st.rerun()
    if tav_c2.button("Clear Tavily Key", key="clear_tav_btn", disabled=not tavily_key_stored, use_container_width=True): delete_tavily_api_key(); st.session_state.tavily_api_key = ""; st.rerun()
    pdf_mode_on = st.session_state.get('pdf_questioning_mode', False)
    web_search_disabled = not tavily_key_stored or pdf_mode_on
    tooltip = "Enable web search." if tavily_key_stored else "Enter Tavily API key first."
    if pdf_mode_on: tooltip = "Disable PDF Questioning Mode first to enable web search."
    web_search_currently_enabled = st.session_state.get('web_search_enabled', False)
    new_web_search_state = st.toggle("Enable Web Search", value=web_search_currently_enabled, key="web_toggle", disabled=web_search_disabled, help=tooltip)
    if new_web_search_state != web_search_currently_enabled:
        st.session_state.web_search_enabled = new_web_search_state; print(f"Web Search Toggled: {'ON' if new_web_search_state else 'OFF'}"); st.rerun()
    if st.session_state.web_search_enabled: st.info("Web Search ON.", icon="🌐")

    st.divider()

    # --- PDF Section ---
    st.subheader("File Upload (PDF Q&A)")
    # (Code omitted for brevity - same as before)
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="file_uploader")
    if uploaded_files != st.session_state.get("uploaded_files", []): st.session_state.uploaded_files = uploaded_files; st.session_state.vector_store = None; st.session_state.pdf_questioning_mode = False; print("New files uploaded, resetting vector store."); st.rerun()
    if provider_name:
        emb_module = get_provider_module(provider_name); emb_key = st.session_state.api_keys.get(provider_name, "")
        needs_emb_check = (st.session_state.embedding_function is None and st.session_state.embedding_checked_provider != provider_name)
        if needs_emb_check:
            print(f"Checking embedding capability for {provider_name}...")
            if emb_module and hasattr(emb_module, "get_embeddings"):
                needs_key = provider_name not in ['ollama'];
                if emb_key or not needs_key:
                     try:
                         with st.spinner(f"Checking embeddings..."): st.session_state.embedding_function = emb_module.get_embeddings(emb_key)
                         if st.session_state.embedding_function: print("Embedding function loaded.")
                         else: print("Embedding function check returned None."); st.caption(f"PDF Q&A N/A (Embeddings unavailable).")
                     except Exception as e: print(f"Error getting embeddings for {provider_name}: {e}"); st.session_state.embedding_function = None; st.caption(f"PDF Q&A N/A (Error checking).")
                else: print(f"Skipping embedding check for {provider_name}: API key required."); st.caption(f"Enter API key for {provider_name} to enable PDF Q&A.")
            else: print(f"Provider {provider_name} does not support embeddings."); st.caption(f"PDF Q&A N/A (Provider lacks support)."); st.session_state.embedding_function = None
            st.session_state.embedding_checked_provider = provider_name
    can_process = bool(st.session_state.uploaded_files) and st.session_state.embedding_function is not None
    if st.button("Process Uploaded Files", key="proc_files_btn", disabled=not can_process):
        with st.spinner("Processing PDFs..."): vs = process_pdfs(st.session_state.uploaded_files, st.session_state.embedding_function)
        st.session_state.vector_store = vs
        if vs: st.success("Files processed successfully!")
        else: st.error("File processing failed.")
        st.rerun()
    web_search_on = st.session_state.get('web_search_enabled', False)
    vector_store_ready = st.session_state.get("vector_store") is not None
    pdf_qa_disabled = not vector_store_ready or web_search_on
    tooltip = "Enable PDF Q&A mode."
    if not vector_store_ready: tooltip = "Upload and process PDF files first."
    elif web_search_on: tooltip = "Disable Web Search first to enable PDF Q&A."
    pdf_mode_currently_enabled = st.session_state.get('pdf_questioning_mode', False)
    new_pdf_mode_state = st.toggle("PDF Questioning Mode", value=pdf_mode_currently_enabled, disabled=pdf_qa_disabled, key="pdf_toggle", help=tooltip)
    if new_pdf_mode_state != pdf_mode_currently_enabled:
        st.session_state.pdf_questioning_mode = new_pdf_mode_state; print(f"PDF Mode Toggled: {'ON' if new_pdf_mode_state else 'OFF'}"); st.rerun()
    if st.session_state.pdf_questioning_mode: st.info("PDF Q&A Mode ON.", icon="📄")

    st.divider()

    # --- Chat Management ---
    st.subheader("Chat Management")
    # (Code omitted for brevity - same as before)
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_sidebar"): create_new_chat(); st.rerun()
    if st.session_state.chats:
        chat_opts = {cid: data["name"] for cid, data in st.session_state.chats.items()}; ids_ordered = sorted(chat_opts.keys(), reverse=True); names_ordered = [chat_opts[cid] for cid in ids_ordered]
        curr_chat_id_in_state = st.session_state.get('current_chat_id')
        curr_name = st.session_state.chats.get(curr_chat_id_in_state, {}).get("name") if curr_chat_id_in_state else None
        try: curr_idx = names_ordered.index(curr_name) if curr_name in names_ordered else 0
        except ValueError: curr_idx = 0
        selected_chat_name_widget_value = st.selectbox("Select Chat", options=names_ordered, index=curr_idx, key="chat_selector")
        selected_chat_id = next((cid for cid, name in chat_opts.items() if name == selected_chat_name_widget_value), None)
        if selected_chat_id and selected_chat_id != curr_chat_id_in_state:
             print(f"Chat CHANGED via Selectbox: {curr_chat_id_in_state} -> {selected_chat_id}"); switch_chat(selected_chat_id)
        active_chat_id_for_buttons = st.session_state.get('current_chat_id')
        if active_chat_id_for_buttons and active_chat_id_for_buttons in st.session_state.chats:
            current_name = st.session_state.chats[active_chat_id_for_buttons]["name"]; new_name = st.text_input("Rename chat", value=current_name, key=f"rename_{active_chat_id_for_buttons}")
            rn_col, dl_col = st.columns(2)
            if rn_col.button("Rename", use_container_width=True, key=f"rn_btn_{active_chat_id_for_buttons}", disabled=not new_name or new_name==current_name): st.session_state.chats[active_chat_id_for_buttons]["name"]=new_name; save_chat(active_chat_id_for_buttons, st.session_state.chats[active_chat_id_for_buttons]); st.success("Renamed."); st.rerun()
            if dl_col.button("Delete Chat", type="primary", use_container_width=True, key=f"dl_btn_{active_chat_id_for_buttons}"):
                del_id = active_chat_id_for_buttons; del st.session_state.chats[del_id]; delete_chat_file(del_id); rem_ids = sorted(st.session_state.chats.keys(), reverse=True); next_id = rem_ids[0] if rem_ids else None; st.warning("Chat deleted."); st.session_state.current_chat_id = next_id
                if next_id: switch_chat(next_id)
                else: create_new_chat()
                st.rerun()
    if st.button("Clear All Chats", type="secondary", use_container_width=True, key="clear_all_chats_btn"):
        for cid in list(st.session_state.chats.keys()): delete_chat_file(cid)
        st.session_state.chats={}; st.session_state.current_chat_id=None; st.session_state.vector_store=None; st.session_state.uploaded_files=[]
        st.session_state.pdf_questioning_mode = False; st.session_state.web_search_enabled = False; st.warning("All chats cleared.");
        st.session_state.selected_provider = AVAILABLE_PROVIDERS[0] if AVAILABLE_PROVIDERS else None; st.session_state.selected_model = None
        update_models_list(); create_new_chat(); st.rerun()

    # --- Advanced Actions ---
    st.subheader("Advanced")
    # (Code omitted for brevity - same as before, including fix for clear messages)
    if st.session_state.current_chat_id and st.session_state.chats.get(st.session_state.current_chat_id, {}).get("messages"):
        with st.expander("Chat Actions & Analytics"):
            current_chat_data_adv = initialize_chat_data(st.session_state.chats[st.session_state.current_chat_id])
            messages_curr_adv = current_chat_data_adv["messages"]
            chat_name_adv = current_chat_data_adv['name']
            c1, c2, c3 = st.columns(3)
            c1.download_button("Export (.txt)", export_chat_to_txt(messages_curr_adv), f"{APP_TITLE}_{chat_name_adv}.txt", "text/plain", use_container_width=True, key="exp_txt")
            c2.download_button("Export (.csv)", export_chat_to_csv(messages_curr_adv), f"{APP_TITLE}_{chat_name_adv}.csv", "text/csv", use_container_width=True, key="exp_csv")
            if c3.button("Clear Msgs", help="Clear messages in this chat", use_container_width=True, key="clr_msgs"):
                system_prompt = current_chat_data_adv.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
                st.session_state.chats[st.session_state.current_chat_id]["messages"] = [Message("system", system_prompt)]
                st.session_state.chats[st.session_state.current_chat_id]["memory"] = None
                save_chat(st.session_state.current_chat_id, st.session_state.chats[st.session_state.current_chat_id])
                update_token_count_indicator() # Update count after clearing
                st.rerun()
            st.markdown("---"); st.markdown("**Analytics**")
            try: analytics = calculate_analytics(messages_curr_adv); a1,a2,a3=st.columns(3); a1.metric("Msgs", analytics["total_messages"]); a2.metric("Words", analytics["word_count"]); a3.metric("User", analytics["user_messages"])
            except Exception as e: st.caption(f"Analytics error: {e}")


# --- Main Chat Area ---
if not st.session_state.current_chat_id:
    st.info("👈 Select or create a chat to get started.")
elif st.session_state.current_chat_id not in st.session_state.chats:
    st.error("Selected chat not found. Switching to latest...")
    st.session_state.current_chat_id = max(st.session_state.chats.keys(), default=None) if st.session_state.chats else None
    if not st.session_state.current_chat_id: create_new_chat()
    time.sleep(0.5); st.rerun()
else:
    current_chat_data = initialize_chat_data(st.session_state.chats[st.session_state.current_chat_id])
    messages_to_display = list(current_chat_data.get("messages", []))
    chat_id_main = st.session_state.current_chat_id

    for i, msg in enumerate(messages_to_display):
        if msg.role == 'system' and i == 0: continue # Skip system message

        with st.chat_message(msg.role):
            st.markdown(msg.content) # Code highlighting is handled by markdown

            # --- Add Copy/Regen buttons to Assistant messages ---
            if msg.role == "assistant":
                _ , btn_container = st.columns([0.85, 0.15])
                with btn_container:
                    copy_col, regen_col = st.columns([1, 1], gap="small")
                    with copy_col:
                        # (Code omitted for brevity - same as before)
                        copy_key = f"copy_md_html_{chat_id_main}_{i}"
                        escaped_content = json.dumps(msg.content)
                        js_function_name = f"cM_{i}"
                        js_success_obj_str = "{streamlit_event:'toast',message:'Copied!',icon:'✅'}"
                        js_error_obj_str = "{streamlit_event:'toast',message:'Copy Fail!',icon:'❌'}"
                        html_code = f"""
                            <script> function {js_function_name}() {{ navigator.clipboard.writeText({escaped_content}).then(() => {{ window.parent.postMessage({js_success_obj_str}, '*') }}).catch(err => {{ console.error('Copy fail:', err); window.parent.postMessage({js_error_obj_str}, '*'); }}); }} </script>
                            <button onclick="{js_function_name}()" title="Copy message" style="background:none;border:none;padding:0;margin:0 2px 0 0; cursor:pointer;font-size:1.0em;color:inherit;line-height:1;display:inline-flex;align-items:center;justify-content:center;height:24px;width:24px;opacity:0.7;"> 📋 </button>
                        """
                        st.components.v1.html(html_code, height=26)
                    with regen_col:
                        if st.button("🔄", key=f"regen_{chat_id_main}_{i}", help="Regenerate response (from last user prompt)"):
                            regenerate_response()

    # Chat Input
    chat_input_disabled = False; reason = ""
    # (Validation logic omitted for brevity - same as before)
    if not st.session_state.current_chat_id: chat_input_disabled = True; reason = "No active chat."
    elif not st.session_state.selected_provider: chat_input_disabled = True; reason = "Select a provider."
    elif not st.session_state.selected_model or any(err in str(st.session_state.selected_model) for err in ["Error", "Enter API Key", "No models", "N/A"]): chat_input_disabled = True; reason = "Select a valid model."
    elif st.session_state.selected_provider not in ['ollama'] and not st.session_state.api_keys.get(st.session_state.selected_provider): chat_input_disabled = True; reason = f"API Key required for {st.session_state.selected_provider}."
    elif st.session_state.pdf_questioning_mode and st.session_state.web_search_enabled: chat_input_disabled = True; reason = "PDF Mode & Web Search cannot be active simultaneously."

    prompt = st.chat_input("Ask TheArt Chat..." + (f" ({reason})" if reason else ""), key="chat_input", disabled=chat_input_disabled)
    if prompt:
        handle_user_input(prompt)

# --- End of app.py ---