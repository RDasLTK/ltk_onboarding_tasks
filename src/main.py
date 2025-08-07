import os
from src.chatbot import ChatbotLangGraph, ChatbotTools, EnvironmentSetup

def initialize_chatbot():
    # Creating a variable to get API Keys from the .env files.
    env_setup = EnvironmentSetup()
    # Getting the tools for the chatbot.
    chatbot_tools = ChatbotTools()
    # Initialize the chatbot with the tools and environment setup
    chatbot = ChatbotLangGraph(tools=chatbot_tools, env_setup=env_setup)

    # Build the LangGraph for the chatbot
    chatbot_graph = chatbot.build_graph(tools=chatbot_tools)
    return chatbot_graph