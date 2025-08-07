#--------------------------------------------------------------------------------------------------------#
#------------ Getting Required Imports and Setting Up Environment Variables------------------------------#
#--------------------------------------------------------------------------------------------------------#

import os
from dotenv import load_dotenv
load_dotenv()

# LangGraph imports
from langchain_groq import ChatGroq
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage

# Tools Libraries
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

# Tavily Imports
from langchain_tavily import TavilySearch

#---------------------------------------------------------------------------------------------------------#
#-----------------------1. Building a simple Chatbot with Groq--------------------------------------------#
#---------------------------------------------------------------------------------------------------------#

class EnvironmentSetup:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")


# Defining the State class
class State(TypedDict):
    messages: Annotated[list, add_messages]
    # Customization of State. A response from the human in the loop tool
    priority_response: str


class ChatbotTools:
    def __init__(self):
        self.tavily_search_tool = TavilySearch(max_results=2)
        self.memory = InMemorySaver()

    def arxiv_search_tool(self):
        """
        This function initializes the Arxiv search tool with a wrapper that limits the number of results and the content length of each document.
        """
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
        arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        return arxiv_tool

    def wikipedia_search_tool(self):
        """
        This function initializes the Wikipedia search tool with a wrapper that limits the number of results and the content length of each document.
        """
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
        return wiki_tool




    # Getting a human in the loop to perform interruptions.
    @tool
    def human_in_the_loop(
        priority_response: str, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """
        A tool that allows human intervention in the chatbot's decision-making process.
        """
        human_response = interrupt({
            "question": "Please verify if Web Based Search needs to be proceeded with"
        })

        if not human_response.strip().lower().startswith('y'):
            priority_response = "Do not proceed with Tavily Search"
            state_update = {
                "messages": [ToolMessage("Human intervention declined", tool_call_id)],
                "priority_response": priority_response
            }
        else:
            # If the human response is affirmative, proceed with the search
            priority_response = "Proceed with Tavily Search"
            state_update = {
                "messages": [ToolMessage("Human intervention accepted", tool_call_id)],
                "priority_response": priority_response
            }
        # Return the state update as a Command
        return Command(update=state_update)


class ChatbotLangGraph():
    def __init__(self, tools: ChatbotTools, env_setup: EnvironmentSetup):
        self.llm = ChatGroq(model_name='gemma2-9b-it', groq_api_key=env_setup.groq_api_key)
        self.llm_and_tools  = self.llm.bind_tools([tools.tavily_search_tool, tools.human_in_the_loop])
        self.graph_builder = StateGraph(State)
        self.memory = tools.memory
        self.tools  = [tools.tavily_search_tool, tools.human_in_the_loop, tools.arxiv_search_tool(), tools.wikipedia_search_tool()]

    def handle_value_error(e: ValueError) -> str:
        """
        This is a custom error handler for ValueError exceptions.
        """
        return f"Error: {str(e)}"

    def chatbot(self, state: State):
        """
        This function defines the chatbot's State
        """
        message = self.llm_and_tools.invoke(state["messages"])
        assert(len(message.tool_calls) <= 1)
        return {"messages": [message]}

    def build_graph(self, tools: ChatbotTools):
        """
        This function builds the LangGraph for the chatbot.
        """
        external_tools = [
            tools.tavily_search_tool,
            tools.human_in_the_loop,
            tools.arxiv_search_tool,
            tools.wikipedia_search_tool
        ]
        tool_node = ToolNode(tools=external_tools,handle_tool_errors=self.handle_value_error())
        self.graph_builder.add_node("Chatbot", self.chatbot)
        self.graph_builder.add_node('tools', tool_node)
        self.graph_builder.add_conditional_edges(
            "Chatbot",
            tools_condition
        )
        self.graph_builder.add_edge("tools", "Chatbot")
        self.graph_builder.add_edge(START, "Chatbot")
        return self.graph_builder.compile(checkpointer=self.memory)





