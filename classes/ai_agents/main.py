# %%
import os
from datetime import datetime, timedelta

import yfinance as yf
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from crewai_tools import CSVSearchTool
from langchain_community.tools import DuckDuckGoSearchResults
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


# %%
@tool(
    "Yahoo Finance Tool",
)
def fetch_stock_price(ticket: str):
    """Fetches stocks prices for {ticket} from the last year about a specific company from the Yahoo Finance API."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    stock = yf.download(
        ticket, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
    )
    return stock


# %%
stocketPriceAnalyst = Agent(
    role="Senior Stock Price Analyst",
    goal="Find the {ticket} stock price and analyses price trends. Compare with the price that the customer paid.",
    backstory="""
    You're a highly experienced in analyzing the price of specific stocks and make predictions about its future price.
    """,
    verbose=True,
    llm=llm,
    max_iter=5,
    allow_delegation=False,
    memory=True,
)

# %%
getStockPrice = Task(
    description="Analyze the stock {ticket} price history and create a price trend analyses of up, down or sideways.",
    expected_output="""
  Specify the current trend stocks price - Up, down or sideways.
  eg. 'stock=AAPL, price UP'
  """,
    tools=[fetch_stock_price],
    agent=stocketPriceAnalyst,
)

# %%
newsAnalyst = Agent(
    role="News Analyst",
    goal="""
    Create a short summary of the market news related to the stock {ticket} company.
    Provide a market Fear & Greed Index score about the company.
    For each requested stock asset, specify a number between 0 ans 100, where 0 is extreme fear and 100 is extreme greed.
    """,
    backstory="""
    You're highly experienced in analyzing market trends and news for more than 10 years.
    You're also a master level analyst in the human psychology.
    You understand the news, their titles and information, but you look at those with a healthy dose of skepticism.
    You consider the source of the news articles.
    """,
    verbose=True,
    llm=llm,
    max_iter=5,
    allow_delegation=False,
    memory=True,
)


# %%
@tool("Search Tool")
def search_news(query: str) -> str:
    """Search for news articles using DuckDuckGo."""
    search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)
    return search_tool.run(query)


# %%
getNews = Task(
    description=f"""
    Use the search tool to search news about the stock ticket.
    The current date is {datetime.now()}
    Compose the results into a helpful report.
    """,
    expected_output="""
    A summary of the overall market and one paragraph summary for the requested asset.
    Include the fear/greed score based on the news. Use the format:
    <STOCK TICKET>
    <SUMMARY BASED ON NEWS>
    <FEAR/GREED SCORE>
    """,
    agent=newsAnalyst,
    tools=[search_news],
)

# %%
