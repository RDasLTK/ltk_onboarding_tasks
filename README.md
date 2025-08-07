# LangGraph Onboarding Tutorial Project - Chatbot using LangGraph

This project demonstrates a chatbot application built using Python, Streamlit, and LangGraph. The chatbot integrates tools such as Tavily Search, Wiki and Arxiv. It also implements Human in the Loop. Finally a streamlit application allows the user to interact with the chatbot.
## Features

- **Conversational Chatbot**: Interact with the chatbot using natural language.
- **Streamlit Integration**: A user-friendly web interface for interaction.
- **Customizable Tools**: Easily extend the chatbot with additional tools and configurations.

## Installation

1. Clone the repository:
   ```bash
   git clone <this repository url>
   cd <your local folder> # ltk_onboarding_tasks

2. Create a virtual environment and activate it:  
```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Set up environment variables:  
   1. Create a .env file in the root directory.
   2. Add your API keys and other configurations as required.

## Usage
1. Run the Streamlit app:  
```bash
streamlit run app/streamlit_app.py
```
2. Open the app in your browser at http://localhost:8501.  
3. Interact with the chatbot and visualize project graphs.