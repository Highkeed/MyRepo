import os
from datetime import datetime
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from dotenv import load_dotenv
load_dotenv('.env')

# 1. Configure the Gemini 2.5 Flash Lite LLM
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY"), # Ensure key is in environment
    temperature=0.7
)

# 2. Define a Tool
@tool("get_current_time")
def get_current_time_tool():
    """Returns the current date and time as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 3. Define Agents
time_specialist = Agent(
    role="Time Specialist",
    goal="Retrieve the exact current time and date.",
    backstory="You are an expert in precision timekeeping.",
    tools=[get_current_time_tool],
    llm=gemini_llm,
    verbose=True
)

summary_writer = Agent(
    role="Communication Specialist",
    goal="Write a friendly, informative message based on the time provided.",
    backstory="You specialize in turning raw data into engaging human-readable summaries.",
    llm=gemini_llm,
    verbose=True
)

# 4. Define Sequential Tasks
# The second task will receive the output of the first task automatically
fetch_time_task = Task(
    description="Retrieve the current date and time using your tools.",
    expected_output="The exact current date and time string.",
    agent=time_specialist
)

format_message_task = Task(
    description="Take the current time provided and write a polite 'Good morning/afternoon' message with a fun fact about time.",
    expected_output="A two-sentence friendly message containing the current time.",
    agent=summary_writer,
    context=[fetch_time_task] # Explicitly passing context from the first task
)

# 5. Assemble and Kickoff the Crew
my_crew = Crew(
    agents=[time_specialist, summary_writer],
    tasks=[fetch_time_task, format_message_task],
    process=Process.sequential, # Tasks execute in the order listed
    verbose=True
)

result = my_crew.kickoff()
print(f"\n\nFINAL OUTPUT:\n{result}")

