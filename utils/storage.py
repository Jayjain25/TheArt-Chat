# --- Start of utils/storage.py ---
import streamlit as st
import os
import json
import pickle # Needed for potential future memory saving, though not currently used for saving
import keyring
from .helpers import Message # Assuming Message class is in helpers.py

# --- Constants ---
DATA_DIR = "data"
CHAT_DIR = os.path.join(DATA_DIR, "chats")
SAVED_MODELS_FILE = os.path.join(DATA_DIR, "saved_models.json")
PROMPT_TEMPLATES_FILE = os.path.join(DATA_DIR, "prompt_templates.json") # Added path for templates
TAVILY_KEY_FILE = os.path.join(DATA_DIR, ".tavily_api_key") # Store Tavily key in data dir too for consistency
KEYRING_SERVICE_NAME = "TheArtChat_APIKeys" # Unique name for keyring service

# --- Ensure Directories Exist ---
os.makedirs(CHAT_DIR, exist_ok=True) # This also creates DATA_DIR if it doesn't exist

# --- Chat Storage ---

def save_chat(chat_id, chat_data):
    """Saves a single chat session to a JSON file, excluding non-serializable objects."""
    if not chat_id or not chat_data:
        print("Warning: Attempted to save chat with invalid chat_id or data.")
        return
    filepath = os.path.join(CHAT_DIR, f"{chat_id}.json")
    try:
        # Create a copy to avoid modifying the original session state dict directly
        serializable_data = chat_data.copy()

        # Convert Message objects to dictionaries
        if "messages" in serializable_data and isinstance(serializable_data["messages"], list):
             # Ensure message objects have the to_dict method
             serializable_data["messages"] = [msg.to_dict() if hasattr(msg, 'to_dict') else vars(msg) for msg in serializable_data["messages"]]
        else:
             serializable_data["messages"] = [] # Ensure messages list exists

        # --- Exclude non-serializable objects ---
        # These will be reconstructed on load or interaction
        serializable_data.pop("memory", None)
        serializable_data.pop("vector_store", None)
        serializable_data.pop("embedding_function", None) # Also exclude this if present

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=4)
    except Exception as e:
        st.error(f"Error saving chat {chat_id}: {e}")
        print(f"Error saving chat {chat_id}: {e}") # Also print to console

def load_chats():
    """Loads all chat sessions from the chat directory, initializing placeholders."""
    chats = {}
    if not os.path.exists(CHAT_DIR):
        print(f"Chat directory not found: {CHAT_DIR}")
        return chats
    try:
        for filename in os.listdir(CHAT_DIR):
            if filename.endswith(".json"):
                chat_id = filename[:-5] # Remove .json extension
                filepath = os.path.join(CHAT_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)

                        # Convert message dictionaries back to Message objects
                        # Ensure Message class has a from_dict class method or suitable constructor
                        chat_data["messages"] = [Message.from_dict(msg) if hasattr(Message, 'from_dict') else Message(**msg) for msg in chat_data.get("messages", [])]

                        # --- Initialize placeholders for non-saved objects ---
                        chat_data["memory"] = None # Needs reconstruction
                        chat_data["vector_store"] = None # Needs reconstruction on demand

                        # --- Load other necessary fields with defaults if missing (for backward compatibility) ---
                        chat_data.setdefault("name", f"Chat_{chat_id[-6:]}") # Default name if missing
                        chat_data.setdefault("provider", None) # Default provider if missing
                        chat_data.setdefault("model", None) # Default model if missing
                        chat_data.setdefault("params", {"temperature": 0.7, "max_tokens": 1024}) # Default params
                        chat_data.setdefault("system_prompt", "You are a helpful assistant.") # Default system prompt
                        chat_data.setdefault("memory_type", "summary_buffer") # Default memory type
                        chat_data.setdefault("memory_config", {"max_token_limit": 600}) # Default memory config
                        # Removed pinned_message_indices initialization

                        chats[chat_id] = chat_data
                except json.JSONDecodeError:
                    st.warning(f"Skipping corrupted chat file: {filename}")
                    print(f"Warning: Skipping corrupted chat file: {filename}")
                except Exception as e:
                    st.error(f"Error loading chat file {filename}: {e}")
                    print(f"Error loading chat file {filename}: {e}")
    except Exception as e:
        st.error(f"Error reading chat directory {CHAT_DIR}: {e}")
        print(f"Error reading chat directory {CHAT_DIR}: {e}")
    return chats

def delete_chat_file(chat_id):
    """Deletes the chat file locally."""
    if not chat_id: return
    filepath = os.path.join(CHAT_DIR, f"{chat_id}.json")
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted chat file: {filepath}")
    except Exception as e:
        st.error(f"Error deleting chat file {filepath}: {e}")
        print(f"Error deleting chat file {filepath}: {e}")

# --- API Key Storage (using keyring) ---

def save_api_key(provider_name, api_key):
    """Saves API key securely using system keyring."""
    try:
        keyring.set_password(KEYRING_SERVICE_NAME, provider_name, api_key)
        print(f"API Key for {provider_name} saved to keyring.")
    except Exception as e:
        st.error(f"Failed to save API key for {provider_name} using keyring: {e}", icon="⚠️")
        st.warning("Keyring might not be configured correctly. Key NOT saved securely.", icon="ℹ️")
        print(f"Keyring Error (save) for {provider_name}: {e}")

def get_api_key(provider_name):
    """Retrieves API key from system keyring. Returns None if not found or error."""
    try:
        key = keyring.get_password(KEYRING_SERVICE_NAME, provider_name)
        # print(f"Retrieved API Key for {provider_name} from keyring: {'Found' if key else 'Not Found'}") # Optional debug
        return key
    except Exception as e:
        # Don't necessarily show error in UI, might be expected if no key saved
        print(f"Keyring Error (get) for {provider_name}: {e}")
        return None

def delete_api_key(provider_name):
     """Deletes API key from system keyring."""
     try:
         keyring.delete_password(KEYRING_SERVICE_NAME, provider_name)
         print(f"Deleted stored API key for {provider_name} from keyring.")
         st.toast(f"Deleted stored API key for {provider_name}.", icon="🗑️")
     except keyring.errors.PasswordDeleteError:
         # This is not an error, just means the key wasn't there
         print(f"No stored API key found for {provider_name} to delete.")
         st.toast(f"No stored API key found for {provider_name} to delete.", icon="ℹ️")
     except Exception as e:
         st.error(f"Failed to delete API key for {provider_name} from keyring: {e}", icon="⚠️")
         print(f"Keyring Error (delete) for {provider_name}: {e}")

# --- Tavily API Key Storage (Using File) ---
# Note: Keyring might be a more secure alternative if consistency is desired.

def save_tavily_api_key(api_key: str) -> None:
    """Saves the Tavily API key to a file in the data directory."""
    try:
        with open(TAVILY_KEY_FILE, "w", encoding='utf-8') as f:
            f.write(api_key)
        print(f"Tavily API Key saved to {TAVILY_KEY_FILE}")
    except Exception as e:
        st.error(f"Error saving Tavily API key: {e}")
        print(f"Error saving Tavily API key: {e}")

def get_tavily_api_key() -> str:
    """Retrieves the Tavily API key from the file. Returns empty string if not found."""
    try:
        if os.path.exists(TAVILY_KEY_FILE):
            with open(TAVILY_KEY_FILE, "r", encoding='utf-8') as f:
                return f.read().strip()
        else:
            return ""
    except Exception as e:
        print(f"Error reading Tavily API key file ({TAVILY_KEY_FILE}): {e}")
        return ""

def delete_tavily_api_key() -> bool:
    """Deletes the saved Tavily API key file."""
    try:
        if os.path.exists(TAVILY_KEY_FILE):
            os.remove(TAVILY_KEY_FILE)
            print(f"Deleted Tavily API key file: {TAVILY_KEY_FILE}")
            st.toast("Deleted stored Tavily API key.", icon="🗑️")
            return True
        else:
            print("No Tavily API key file found to delete.")
            st.toast("No stored Tavily API key found to delete.", icon="ℹ️")
            return False
    except Exception as e:
        st.error(f"Error deleting Tavily API key file: {e}")
        print(f"Error deleting Tavily API key file: {e}")
        return False

# --- Saved Models Storage ---

def load_saved_models():
    """Loads manually saved models from a JSON file."""
    if not os.path.exists(SAVED_MODELS_FILE):
        return {}
    try:
        with open(SAVED_MODELS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Saved models file not found or corrupted at {SAVED_MODELS_FILE}. Returning empty. Error: {e}")
        return {} # Return empty dict if file corrupt or not found
    except Exception as e:
        st.error(f"Error loading saved models file: {e}")
        print(f"Error loading saved models file: {e}")
        return {}

def save_saved_models(saved_models_dict):
    """Saves the dictionary of manually saved models to a JSON file."""
    try:
        with open(SAVED_MODELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(saved_models_dict, f, indent=4)
            print(f"Saved models file updated at {SAVED_MODELS_FILE}")
    except Exception as e:
        st.error(f"Error saving models file: {e}")
        print(f"Error saving models file: {e}")


# --- Prompt Templates Storage ---

def load_prompt_templates():
    """Loads prompt templates from a JSON file."""
    if os.path.exists(PROMPT_TEMPLATES_FILE):
        try:
            with open(PROMPT_TEMPLATES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading prompt templates file ({PROMPT_TEMPLATES_FILE}): {e}")
            st.warning(f"Could not load prompt templates: {e}")
            return {} # Return empty dict on error
    print("Prompt templates file not found, returning empty.")
    return {} # Return empty dict if file doesn't exist

def save_prompt_templates(templates):
    """Saves prompt templates to a JSON file."""
    try:
        # Ensure the data directory exists (redundant if handled above, but safe)
        os.makedirs(os.path.dirname(PROMPT_TEMPLATES_FILE), exist_ok=True)
        with open(PROMPT_TEMPLATES_FILE, 'w', encoding='utf-8') as f:
            json.dump(templates, f, indent=4) # Save with indentation
        print(f"Prompt templates saved to {PROMPT_TEMPLATES_FILE}")
    except IOError as e:
        print(f"Error saving prompt templates file ({PROMPT_TEMPLATES_FILE}): {e}")
        st.error(f"Failed to save prompt templates: {e}") # Show error in UI

# --- End of utils/storage.py ---