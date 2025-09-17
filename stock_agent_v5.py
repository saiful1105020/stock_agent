"""
The agent now integrates Human in the loop.
Improvements over stock_agent_v4.py:
    - Add persistent memory
    - Add human in the loop
"""
import os, sys
import getpass

# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain_core.tools import tool
from langgraph.types import Command, interrupt

from typing_extensions import TypedDict
from IPython.display import Image, display

# Schema for structured output
from pydantic import BaseModel, Field

from uuid import uuid4
from langgraph.checkpoint.memory import InMemorySaver

SELF_THOUGHT=1
EXPLICIT=2

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

#_set_env("ANTHROPIC_API_KEY")
_set_env("OPENAI_API_KEY")
_set_env("SERPER_API_KEY")

# Graph state
class State(TypedDict):
    messages: list[str]
    list_of_companies: list[str]
    amounts_to_invest: list[float]
    risk_levels: list[str]
    justifications: list[str]
    follow_up_required: bool
    follow_up_question: str

# Structured output schema for the final output
class InitialStockAdvice(BaseModel):
    summary: str = Field(
        None, description="Plain text response containing initial greeting and list of companies."
    )
    list_of_companies: list[str] = Field(
        None, description="List of companies to invest in."
    )

# Follow-up question decision
class IsFollowUpNeeded(BaseModel):
    decision: bool = Field(
        None, description="Do we need more information from the user?"
    )
    response: str = Field(
        None, description="Summary response to user input."
    )
    question: str = Field(
        None, decision="Follow up question to ask the user."
    )

# Structured output schema for the final output
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
# Base LLM
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
llm = ChatOpenAI(model="gpt-4o")

# LLM with structured output for the final summary
structured_llm = llm.with_structured_output(StockAdvice)
structured_initial_llm = llm.with_structured_output(InitialStockAdvice)
follow_up_llm = llm.with_structured_output(IsFollowUpNeeded)

# LLM with tools (e.g., Google Search)
google_search_wrapper = GoogleSerperAPIWrapper(serper_api_key=os.environ["SERPER_API_KEY"])
@tool()
def websearch(query: str) -> str:
    """
    Use Google Search to find relevant information.
    """
    return google_search_wrapper.run(query)
tools = [websearch]
llm_with_tools = llm.bind_tools(tools)
    
### LLM Agent Nodes ###
def initial_response(state: State):
    """Initial LLM node to greet the user."""
    messages = state.get("messages", [])
    msg = structured_initial_llm.invoke(messages)
    return {"list_of_companies":msg.list_of_companies, "messages": messages + [AIMessage(content=msg.summary+"\n"+",".join(msg.list_of_companies), id=SELF_THOUGHT)]}

def verify_companies(state: State):
    """LLM node to verify the list of companies using web search."""
    messages = state.get("messages", [])
    # Append a system message to instruct the LLM to verify the companies
    new_msg = SystemMessage(content="Verify the list of companies mentioned in the previous message. Use web search if needed. If any company is not valid, remove it from the list. Summarize this update.")
    messages.append(new_msg)
    msg = llm_with_tools.invoke(messages)
    messages.append(
        AIMessage(content=msg.content, tool_calls=msg.tool_calls)
    )

    for call in msg.tool_calls:
        if call["name"] == "websearch":
            tool_result = websearch.invoke(call["args"])
            
            # Append tool results
            messages.append(
                ToolMessage(content=tool_result, tool_call_id=call["id"])
            )
        else:
            assert False, "Undefined tool calls"

    msg = structured_initial_llm.invoke(messages)
    return {"list_of_companies":msg.list_of_companies, "messages": messages + [AIMessage(content=msg.summary+"\n"+",".join(msg.list_of_companies), id=EXPLICIT)]}

def collect_user_feedback(state: State):
    messages = list(state.get("messages", []))
    last_ai_message = messages[-1]
    if hasattr(last_ai_message, "content"):
        value = interrupt({"question": last_ai_message.content})
    elif hasattr(last_ai_message, "question"):
        value = interrupt({"question": last_ai_message.response + "\n" + last_ai_message.question})
    
    messages.append(HumanMessage(content=value))
    new_msg = SystemMessage(content="Do you want to ask follow-up question?")
    messages.append(new_msg)
    msg = follow_up_llm.invoke(messages)
    return {"follow_up_required":msg.decision, "follow_up_question": msg.question, "messages": messages + [AIMessage(content=msg.response + "\n" + msg.question, id=EXPLICIT)]}

def followup_router(state: State):
    if state["follow_up_required"]:
        return "FOLLOW_UP"
    else:
        return "UPDATE"
    
def update_list_of_companies(state: State):
    """
    LLM node to redo the company listing based on user response
    """
    messages = state.get("messages", [])
    new_msg = SystemMessage(content="Incorporate user input, and revise your recommended list of companies.")
    messages.append(new_msg)
    msg = structured_initial_llm.invoke(messages)
    return {"messages": messages + [AIMessage(content=msg.summary+"\n"+",".join(msg.list_of_companies), id=EXPLICIT)]} 

def assign_budget(state: State):
    """LLM node to acknowledge user's budget."""
    messages = state.get("messages", [])
    new_msg = SystemMessage(content="Acknowledge the user's budget. Split the budget across the stocks you have recommended. Make sure the total is equal to the user's budget.")
    messages.append(new_msg)
    msg = llm.invoke(messages)
    return {"messages": messages + [AIMessage(content=msg.content, id=SELF_THOUGHT)]}

def guardrail_check(state: State):
    """LLM node to ensure the response adheres to guardrails."""
    messages = state.get("messages", [])
    new_msg = SystemMessage(content="Explicitly mention that you are not a financial advisor and this is not financial advice. Briefly clarify the risks involved in stock investments. Mention that the user should do their own research before making any investment decisions.")
    messages.append(new_msg)
    msg = llm.invoke(messages)
    return {"messages": messages + [AIMessage(content=msg.content, id=SELF_THOUGHT)]}

def final_summary(state: State):
    """LLM node to provide a final summary of the investment advice."""
    messages = state.get("messages", [])
    new_msg = SystemMessage(content="Provide a final summary of the investment advice in a structured format, including the cautions.")
    messages.append(new_msg)
    msg = structured_llm.invoke(messages)
    return {"list_of_companies": msg.list_of_companies, "amounts_to_invest": msg.amounts_to_invest, "risk_levels": msg.risk_levels, "justifications": msg.justification, "messages": messages + [AIMessage(content=str(msg.summary), id=EXPLICIT)]}

def build_graph():
    """Builds the state graph for the stock recommendation agent."""
    builder = StateGraph(State)
    builder.add_node("initial_response", initial_response)
    builder.add_node("verify_companies", verify_companies)
    builder.add_node("collect_user_feedback", collect_user_feedback)
    builder.add_node("update_list_of_companies", update_list_of_companies)
    builder.add_node("assign_budget", assign_budget)
    builder.add_node("guardrail_check", guardrail_check)
    builder.add_node("final_summary", final_summary)
    
    builder.add_edge(START, "initial_response")
    builder.add_edge("initial_response", "verify_companies")
    builder.add_edge("verify_companies", "collect_user_feedback")
    builder.add_conditional_edges("collect_user_feedback", followup_router, {"FOLLOW_UP":"collect_user_feedback", "UPDATE":"update_list_of_companies"})
    builder.add_edge("update_list_of_companies", "assign_budget")
    builder.add_edge("assign_budget", "guardrail_check")
    builder.add_edge("guardrail_check", "final_summary")
    builder.add_edge("final_summary", END)

    memory = InMemorySaver()
    return builder.compile(checkpointer=memory)

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

    # Kick off
    # Initial messages
    system_message = SystemMessage(content="You are a financial advisor. Provide stock investment advice based on user's budget.")
    query = HumanMessage(content="I have 100$ to invest. Which stocks should I buy?")
    initial = {
        "messages": [system_message, query]
    }

    # Run to first interrupt
    thread = {"configurable": {"thread_id": f"demo-{uuid4()}"}} 
    for event in graph.stream(initial, thread, stream_mode="updates"):
        # You can inspect events here if you like
        # print(event)
        pass

    while True:
        for event in graph.stream(None, config=thread, stream_mode="updates"):
            # If the graph hits an interrupt, you'll see it here
            if isinstance(event, dict) and "__interrupt__" in event:
                question = event["__interrupt__"][0].value["question"]
                print(question)
                ans = input("> ").strip()

                # Resume from the same run
                for _ in graph.stream(Command(resume=ans), config=thread, stream_mode="updates"):
                    pass
                break  # break inner for-loop, continue while-loop
        else:
            # no interrupt yielded and graph reached END
            break


    # Save a reference to the original stdout
    original_stdout = sys.stdout

    # ---- IMPORTANT: read the persisted state, not the original dict ----
    output = graph.get_state(thread).values
    # print(output)
    
    # Open the file in write mode ('w')
    with open('output.txt', 'w') as f:
        # Redirect stdout to the file
        sys.stdout = f

        # Print the final structured output
        print("\nFinal Structured Output:")
        print(f"List of Companies: {output.get('list_of_companies')}")
        print(f"Amounts to Invest: {output.get('amounts_to_invest')}")
        print(f"Risk Levels: {output.get('risk_levels')}")
        print(f"Justification: {output.get('justifications')}")

        print("### Message History ###")

        # Print the entire conversation
        messages = output.get("messages", [])
        for msg in messages:
            msg.pretty_print()

    # Restore original stdout
    sys.stdout = original_stdout

    # Print the conversation
    messages = output.get("messages", [])
    for msg in messages:
        # Skip printing system messages and inner thoughts
        if (msg.type != "system") and (int(msg.id or 0) != SELF_THOUGHT):
            msg.pretty_print()
