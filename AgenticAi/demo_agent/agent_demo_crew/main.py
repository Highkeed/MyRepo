from crewai import Agent, Task, Crew, Process
from tools.YFinanceStockAnalysisTool import YFinanceStockAnalysisTool
from crewai_tools import FileReadTool
import time
from dotenv import load_dotenv

# Set override=True to force usage of the specific file
load_dotenv('c:/codellm/.env', override=True)

start_time = time.time()

yfinance_stock_data_tool = YFinanceStockAnalysisTool()
file_read_tool = FileReadTool(file_path='warren_buffet_investement_principles.txt')

# from langchain_groq import ChatGroq
# AGENT_MOODEL = ChatGroq(model="groq/qwen/qwen3-32b") #"groq/llama-3.3-70b-versatile"

from crewai import LLM
AGENT_MOODEL = LLM(model="gemini/gemini-2.5-flash")  #"gemini/gemini-3-pro-preview"

# Agent definitions
stock_research_manager = Agent(
    role='Stock Research Manager',
    verbose=True,
    goal='Gather relevant stock news and data',
    backstory="""
        You are a methodical manager supervising stock research.
        You prevent process loops and maintain high-quality output.
        """,
    tools=[],
    memory=True,
    llm=AGENT_MOODEL
)

senior_stock_analyst = Agent(
    role='Senior Stock Analyst',
    goal='Analyze the provided stock data and news to give investment advice',
    backstory="""
        You are an AI-powered senior analyst applying Warren Buffett's 
        investment principles. You are very strict with your recommendations 
        and only recommend the best stocks.
        """,
    tools=[],
    memory=True,
    llm=AGENT_MOODEL,
    verbose=True,
    allow_delegation=False
)

# Task definitions
get_stock_analysis_for_ticker = Task(
    description="""
        for a given ticker: {ticker} fetch the data from yfinance and return data to 
        the next agent
    """,
    expected_output="""
        Ticker Kpis:
                - Ticker Symbol
                - Company Name
                - Current Price
                - 52-Week High
                - 52-Week Low
                - Market Cap
                - P/E Ratio
                - P/B Ratio
                - Debt-to-Equity Ratio
                - Current Ratio
                - Dividend Yield (%)
                - 5-Year Revenue Growth Rate (%)
                - Free Cash Flow
                - Profit Margin
                - Operating Margin
                - Earnings Growth
                - Revenue Growth
                - Analyst Target Price
                - Beta
    """,
    tools=[yfinance_stock_data_tool],
    agent=stock_research_manager
)

analyse_ticker = Task(
    description="""
        first you get warren buffett's investment principles by using the 
        file read tool
        then you provide a comprehensive analysis of the ticker
        write short and concise the pros why warren buffett would invest in
        this company and the cons why he wouldn't
        then summarize the company evaluation and provide a recommendation
        then you give a warren buffet buy recomendation from 0 to 10 (10 is best)
    """,
    expected_output="""
        - Pros why Warren Buffett would invest in this company
        - Cons why Warren Buffett wouldn't invest in this company
        - Company Evaluation Summary
        - Warren Buffet Buy Recommendation
    """,
    tools=[file_read_tool],
    agent=senior_stock_analyst,
    output_file='stock_analysis_IDFCFIRST.md'
)


# Crew definition
stock_analysis_crew = Crew(
    agents=[stock_research_manager, senior_stock_analyst],
    tasks=[get_stock_analysis_for_ticker, analyse_ticker],
    process=Process.sequential
)

inputs = {
        "ticker": "INFY.NS"  #IDFCFIRST Bank ticker
}

# Kickoff the process
result = stock_analysis_crew.kickoff(inputs=inputs)
print("Crew usage", stock_analysis_crew.usage_metrics)
# print(result)

print("Time taken: ", time.time() - start_time)
