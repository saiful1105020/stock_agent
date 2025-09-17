import os
import getpass
import pprint

# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Schema for structured output
from pydantic import BaseModel, Field

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

#_set_env("ANTHROPIC_API_KEY")
_set_env("OPENAI_API_KEY")

class StockAdvice(BaseModel):
    list_of_companies: list[str] = Field(None, description="List of companies to invest in.")
    amounts_to_invest: list[float] = Field(
        None, description="Corresponding amounts to invest in each company."
    )
    risk_level: str = Field(None, description="Risk level of the investment strategy.")
    justification: str = Field(
        None, description="Justification for the investment advice."
    )

    def __str__(self):
        return (
            f"List of Companies: {self.list_of_companies}\n"
            f"Amounts to Invest: {self.amounts_to_invest}\n"
            f"Risk Level: {self.risk_level}\n"
            f"Justification: {self.justification}\n"
        )


if __name__ == "__main__":
    # llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
    llm = ChatOpenAI(model="gpt-4o")

    # Augment the LLM with schema for structured output
    structured_llm = llm.with_structured_output(StockAdvice)

    query = "I have 100$ to invest. Which stocks should I buy?"
    output = structured_llm.invoke(query)
    pprint.pprint(output)