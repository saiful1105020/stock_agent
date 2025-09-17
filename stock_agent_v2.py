"""
A more complex agent that uses a state graph to manage conversation state and provide stock investment advice.
Improvements over stock_agent_v1.py:
- Uses a state graph to manage conversation flow.
- Provides final structured output at the end.
- Visualizes the state graph and saves it as a PNG file.
"""
import os
import getpass

# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from typing_extensions import TypedDict
from IPython.display import Image, display

# Schema for structured output
from pydantic import BaseModel, Field

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

#_set_env("ANTHROPIC_API_KEY")
_set_env("OPENAI_API_KEY")

# Graph state
class State(TypedDict):
    messages: list[str]
    list_of_companies: list[str]
    amounts_to_invest: list[float]
    risk_levels: list[str]
    justifications: list[str]

# Structured output schema
class StockAdvice(BaseModel):
    list_of_companies: list[str] = Field(
        None, description="List of companies to invest in."
    )
    amounts_to_invest: list[float] = Field(
        None, description="Corresponding amounts to invest in each company."
    )
    risk_levels: list[str] = Field(
        None, description="Risk levels of investing in each company."
    )
    justification: str = Field(
        None, description="Justification for the investment advice."
    )
    summary: str = Field(
        None, description="A brief summary of the investment advice."
    )

    def __str__(self):
        return (
            f"List of Companies: {self.list_of_companies}\n"
            f"Amounts to Invest: {self.amounts_to_invest}\n"
            f"Risk Levels: {self.risk_levels}\n"
            f"Justification: {self.justification}\n"
            f"Summary: {self.summary}\n"
        )
    
### Define the LLMs we will use ###
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(StockAdvice)
    
### LLM Agent Nodes ###
def stock_recommender(state: State):
    """LLM node to recommend stocks based on user's budget."""
    messages = state.get("messages", [])
    msg = structured_llm.invoke(messages)
    return {"list_of_companies": msg.list_of_companies, "amounts_to_invest": msg.amounts_to_invest, "risk_levels": msg.risk_levels, "justifications": msg.justification, "messages": messages + [AIMessage(content=str(msg.summary))]}

def build_graph():
    """Builds the state graph for the stock recommendation agent."""
    builder = StateGraph(State)
    builder.add_node("stock_recommender", stock_recommender)
    builder.add_edge(START, "stock_recommender")
    builder.add_edge("stock_recommender", END)
    return builder.compile()

if __name__ == "__main__":
    # Build the graph
    graph = build_graph()

    # Display the graph
    img = Image(
            graph.get_graph().draw_mermaid_png(
            )
        )
    with open("block_diagram.png", "wb") as png:
        png.write(img.data)

    # Initial messages
    system_message = SystemMessage(content="You are a financial advisor. Provide stock investment advice based on user's budget.")
    query = HumanMessage(content="I have 100$ to invest. Which stocks should I buy?")
    
    # Invoke the graph
    output = graph.invoke({"messages": [system_message, query]})

    # Print the conversation
    messages = output.get("messages", [])
    for msg in messages:
        msg.pretty_print()

    # Print the final structured output
    print("\nFinal Structured Output:")
    print(f"List of Companies: {output.get('list_of_companies')}")
    print(f"Amounts to Invest: {output.get('amounts_to_invest')}")
    print(f"Risk Levels: {output.get('risk_levels')}")
    print(f"Justification: {output.get('justifications')}")