from tavily import TavilyClient
import streamlit as st

def get_available_models(api_key):
    """Return available search types."""
    return ["basic", "analyze"]

def generate_response(api_key, model_name, messages, params=None):
    """Generate response using Tavily web search."""
    if not api_key:
        return "Please provide a Tavily API key."
    
    try:
        client = TavilyClient(api_key)
        
        # Get the last user message
        last_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), None)
        if not last_user_msg:
            return "No user query found."

        # Perform search based on model type
        search_type = model_name  # basic or analyze
        response = client.search(
            query=last_user_msg,
            search_depth=search_type,
            max_results=5
        )

        # Format the response
        formatted_response = "🔍 Web Search Results:\n\n"
        for result in response.get('results', []):
            formatted_response += f"### [{result['title']}]({result['url']})\n"
            formatted_response += f"{result['content']}\n\n"
            
        return formatted_response

    except Exception as e:
        return f"Error performing web search: {str(e)}"

def get_embeddings(api_key):
    """Tavily doesn't provide embeddings."""
    return None