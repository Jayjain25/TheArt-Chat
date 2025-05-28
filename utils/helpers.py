import streamlit as st
import datetime
import pandas as pd
from io import StringIO
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class Message:
    """Represents a message in the chat history."""
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_langchain(self):
        """Converts the message to a LangChain message object."""
        if self.role == "user":
            return HumanMessage(content=self.content)
        elif self.role == "assistant":
            return AIMessage(content=self.content)
        elif self.role == "system":
            return SystemMessage(content=self.content)
        else: # Fallback for unknown roles
            # Langchain memory usually expects user/ai pairs, system is often handled separately
            print(f"Warning: Unknown message role '{self.role}' encountered during LangChain conversion.")
            # Represent as Human message with role prefix? Or raise error?
            return HumanMessage(content=f"[{self.role.upper()}] {self.content}")


    def to_dict(self):
        """Converts the message to a dictionary for serialization."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data):
        """Creates a Message object from a dictionary."""
        return cls(role=data.get("role", "unknown"), content=data.get("content", ""))

def get_timestamp_id():
    """Generates a unique timestamp-based ID."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def export_chat_to_txt(messages):
    """Exports chat messages to a simple text format."""
    chat_str = ""
    for msg in messages:
        chat_str += f"{msg.role.capitalize()}: {msg.content}\n\n"
    return chat_str

def export_chat_to_csv(messages):
    """Exports chat messages to a CSV format."""
    df = pd.DataFrame([msg.to_dict() for msg in messages])
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def calculate_analytics(messages):
    """Calculates basic chat analytics."""
    analytics = {"total_messages": 0, "word_count": 0, "user_messages": 0, "assistant_messages": 0}
    if not messages: return analytics # Handle empty list

    analytics["total_messages"] = len(messages)
    for msg in messages:
        analytics["word_count"] += len(str(msg.content).split()) # Ensure content is string
        if msg.role == "user":
            analytics["user_messages"] += 1
        elif msg.role == "assistant":
            analytics["assistant_messages"] += 1
    return analytics