# pip install crewai[google-genai], python-dotenv

import os
from datetime import datetime
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from dotenv import load_dotenv

load_dotenv(".env", override=True)

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# 2. Define a tool using the @tool decorator
@tool("get_current_time")
def get_current_time_tool():
    """Returns the current date and time as a string. Use this when the user asks for the time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 3. Define the Agent
time_assistant = Agent(
    role="Time Specialist",
    goal="Provide accurate current time information to the user.",
    backstory="You are a precise timekeeping assistant with access to system clocks.",
    tools=[get_current_time_tool],
    llm=gemini_llm,
    verbose=True
)

# 4. Define the Task
time_task = Task(
    description="Find out the current time and tell the user.",
    expected_output="A friendly sentence stating the exact current date and time.",
    agent=time_assistant
)

# 5. Assemble and Kickoff the Crew
crew = Crew(
    agents=[time_assistant],
    tasks=[time_task],
    process=Process.sequential
)

result = crew.kickoff()
print(f"\n\n########################\n## AGENT OUTPUT:\n########################\n{result}")

