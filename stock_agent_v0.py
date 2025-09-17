import os
import getpass
import pprint

# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

#_set_env("ANTHROPIC_API_KEY")
_set_env("OPENAI_API_KEY")

if __name__ == "__main__":
    # llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
    llm = ChatOpenAI(model="gpt-4o")

    query = "I have 100$ to invest. Which stocks should I buy?"
    output = llm.invoke(query)
    pprint.pprint(output.content)