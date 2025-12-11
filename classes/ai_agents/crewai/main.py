# %%
import os
from datetime import datetime, timedelta

import yfinance as yf
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from crewai_tools import CSVSearchTool
from IPython.display import Markdown
from langchain_community.tools import DuckDuckGoSearchResults

os.environ["OPENAI_API_KEY"] = "NA"


# %%
llm = LLM(
    model="ollama/llama3.2:3b",
)


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
    Provide if the stock is in the customer wallet and if it is, provide the mean price he paid and the total numbers of stocks owned.
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
stockPriceAnalyst = Agent(
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
    agent=stockPriceAnalyst,
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
stockRecommender = Agent(
    role="Chief Stock Analyst",
    goal="Get the data from the customer currently stocks, the provided input of stock price trends and the stock news to provide a recommendation: Buy, Sell or Hold the stock.",
    backstory="""
    You're the leader of the stock analyst team.
    You have a great performance in the past 20 years in stock recommendation.
    With all of your team information, you are able to provide the best recommendation for the customer to achieve the maximum value creation.
    """,
    verbose=True,
    llm=llm,
    max_iter=5,
    allow_delegation=True,
    memory=True,
)

# %%
recommendStock = Task(
    description="""
    Use the stock price trend, the stock news report and the customers stock mean price of the {ticket} to provide a recommendation: Buy, Sell or Hold.
    If the previous reports are not well provided you can delegate back to the specific analyst to work again in the their task.
    """,
    expected_output="""
    A brief paragraph with the summary of the reasons for recommendation and the recommendation itself in one of the three possible outputs: Buy, Sell or Hold.
    Use the format:
    <SUMMARY OF REASONS>
    <RECOMMENDATION>
    """,
    agent=stockRecommender,
    context=[getNews, getStockPrice, getCustomerWallet],
)

# %%
copywriter = Agent(
    role="Stock Content Writer",
    goal="""
    Write an insightful and compelling and informative 4 paragraph long newsletter.
    Based on the stock price report, the news report and the recommendation report.
    """,
    backstory="""
    You are an unbelievable copywriter that understand complex financial concepts and explain for a dummy audience.
    You create compelling stories and narrative that resonate with the audience.
    """,
    verbose=True,
    llm=llm,
    max_iter=5,
    allow_delegation=False,
    memory=True,
)

# %%
writeNewsletter = Task(
    description="""
    Use the stock price trend, the stock news report and stock recommendationto write an insightful and compelling and informative 4 paragraph long newsletter.
    Focus on the stock price trend, news, fear/greed score and the summary reason for the recommendation.
    Include the recommendation in the newsletter.
    """,
    expected_output="""
    An eloquent 4 paragraph newsletter formatted as Markdown in an easy readable manner.
    It should contain:
    - Introduction - set the overrall picture
    - Main part - provides the meat of the analysis including stock price trend, the news, the fear/greed score and the summary reason for the recommendation
    - 3 bullets of the main summary reason of the recommendation
    - Recommendation Summary
    - Recommendation itself
    """,
    agent=copywriter,
    context=[getStockPrice, getNews, recommendStock],
)

# %%
crew = Crew(
    agents=[
        customerManager,
        stockPriceAnalyst,
        newsAnalyst,
        stockRecommender,
        copywriter,
    ],
    tasks=[getCustomerWallet, getStockPrice, getNews, recommendStock, writeNewsletter],
    verbose=True,
    process=Process.hierarchical,
    share_crew=False,
    manager_llm=llm,
)

# %%
results = crew.kickoff(inputs={"ticket": "AMZN"})

# %%
Markdown(results.raw)

# %%
