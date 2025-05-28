import streamlit as st
import requests
from typing import List, Dict, Any
import numpy as np
from langchain_core.embeddings import Embeddings

class NvidiaProvider:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.nvcf.nvidia.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def get_available_models(self) -> List[str]:
        """Return list of available NVIDIA NIM models."""
        return [
            "mixtral-8x7b",
            "llama2-70b",
            "mistral-7b",
            "nemotron-3-8b",
            "yi-34b"
        ]

    def generate_response(self, messages: List[Dict[str, str]], model: str, **params) -> str:
        """
        Generate a response using the NVIDIA NIM API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name to use
            **params: Additional parameters like temperature, max_tokens, etc.
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                **params
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        model_info = {
            "mixtral-8x7b": {
                "description": "Mixtral 8x7B model",
                "context_length": 32768,
                "parameters": "47B"
            },
            "llama2-70b": {
                "description": "Llama 2 70B model",
                "context_length": 4096,
                "parameters": "70B"
            },
            "mistral-7b": {
                "description": "Mistral 7B model",
                "context_length": 32768,
                "parameters": "7B"
            },
            "nemotron-3-8b": {
                "description": "Nemotron-3 8B model",
                "context_length": 16384,
                "parameters": "8B"
            },
            "yi-34b": {
                "description": "Yi 34B model",
                "context_length": 4096,
                "parameters": "34B"
            }
        }
        return model_info.get(model_name, {"description": "Model information not available"})

class NIMEmbeddings(Embeddings):
    """NVIDIA NIM Embeddings wrapper for LangChain."""
    
    def __init__(self, api_key: str):
        """Initialize with NVIDIA NIM API key."""
        self.api_key = api_key
        self.base_url = "https://api.nvcf.nvidia.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Using a suitable embedding model from NVIDIA NIM
        self.model = "yi-34b"  # Can be adjusted based on available embedding models

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        try:
            # Split into batches if needed (NVIDIA might have input limits)
            batch_size = 8
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                payload = {
                    "model": self.model,
                    "input": batch
                }
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                embeddings = [data["embedding"] for data in response.json()["data"]]
                all_embeddings.extend(embeddings)
            
            return all_embeddings
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            raise e

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        try:
            payload = {
                "model": self.model,
                "input": [text]
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            st.error(f"Error generating query embedding: {str(e)}")
            raise e

def get_embeddings(api_key: str) -> Embeddings:
    """Returns an Embeddings instance for use with LangChain."""
    return NIMEmbeddings(api_key=api_key)