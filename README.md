# TheArt-Chat
# Multi-LLM Chat App
## Overview

This Streamlit application provides a unified interface for interacting with various Large Language Models (LLMs). Enter your API key for a supported model, select the model, and start chatting. The goal is to offer a seamless experience for experimenting and working with different LLMs in one place.

## Features

### Current Features

* **Multi-LLM Support:** Chat with a variety of LLMs including:
    * Cohere
    * Google
    * Groqcloud
    * Huggingface
    * Hyperbolic
    * Nvidia NIM
    * Ollama
    * OpenRouter
    * Tavila
    * Together
* **API Key Integration:** Securely enter and manage your API keys for different LLMs directly within the application.
* **Conversation History:** Keep track of your chat conversations.
* **Model Parameter Settings:** Customize LLM behavior with adjustable parameters like temperature and max tokens.
* **Save/Export Chats:** Save or export your chat sessions for future reference.
* **Response Management:**
    * Copy responses with ease.
    * Select from predefined response templates.
    * Save your own custom response templates.
* **Chat with PDF:** Interact with your documents by uploading PDFs and chatting with their content.

### Coming Soon

* **Proper Web Search Integration:** Enhance LLM responses with real-time web search capabilities.
* **Text-to-Speech:** Convert LLM responses into spoken audio.
* **Speech-to-Text:** Input your queries using voice.

## Prerequisites

Before running the application, ensure you have the following:

* **Python:** Python 3.9+ is recommended. You can download it from [python.org](https://www.python.org/downloads/).
* **API Keys:** Obtain API keys from the respective LLM providers you wish to use (e.g., Google Cloud, Hugging Face, OpenAI, etc.).
* **Dependencies:** All required Python libraries are listed in `requirements.txt`.

## Getting Started

Follow these steps to get your Multi-LLM Chat App up and running on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```
    *(Replace `YourUsername` and `YourRepoName` with your actual GitHub username and repository name.)*

2.  **Install dependencies:**
    It's recommended to create a virtual environment first:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
    Then, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py # Or whatever your main Streamlit file is called (e.g., main.py)
    ```
    *(Make sure to replace `app.py` with the actual name of your Streamlit application's main file.)*

4.  **Enter API Keys:**
    Once the application launches in your web browser, you will be prompted to enter your API keys for the desired LLMs.

## Usage

* Select the LLM model you wish to chat with from the dropdown.
* Enter your corresponding API key.
* Start typing your questions or prompts in the chat input area.
* Explore the settings to adjust model parameters or manage templates.
* Use the "Chat with PDF" feature to upload and interact with your documents.

## Contributing

We welcome contributions! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

*(Add your chosen license here, e.g., MIT License, Apache 2.0, etc.)*

---
