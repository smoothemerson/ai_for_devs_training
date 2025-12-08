# %%
import os

from crewai import Agent, Crew, Process, Task
from crewai_tools import CSVSearchTool
from langchain_ollama import ChatOllama

os.environ["OPENAI_API_KEY"] = "NA"


# %%
llm = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")

# %%
csvWalletTool = CSVSearchTool(
    csv="./dataset/wallet.csv",
    config={
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.2:3b",
            },
        },
        "embedding_model": {
            "provider": "ollama",
            "config": {
                "model_name": "nomic-embed-text:v1.5",
            },
        },
    },
)

# %%
customerManager = Agent(
    role="Customer Stocks Manager",
    goal="Get the customer question about the stock {ticket} and search the customer wallet CSV file for the stocks",
    backstory="""
    You're the manager of the customer investiments wallet.
    You are the client first contact and you provide the other analysts with the necessary stock ticket and wallet information.
    """,
    verbose=True,
    llm=llm,
    max_iter=5,
    tools=[csvWalletTool],
    allow_delegation=False,
    memory=True,
)

# %%
getCustomerWallet = Task(
    description="""
    Use the customer question and find the {ticket} in the CSV file.
    Provide if the stock is in the customer wallet and if it is, provide the mean prive he paid and the total numbers of stocks owned.
    """,
    expected_output="If the customer owns the stock, provide the mean price paid and the total stock numbers.",
    agent=customerManager,
)
