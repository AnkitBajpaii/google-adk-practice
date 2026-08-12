import os
import sys
import json
import asyncio
import random
import string
import requests
from uuid import uuid4
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.events import Event
from google.adk.runners import Runner
import google.adk as adk
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
from google.genai.types import Content, Part
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

# --- Initialize our Session Service ---
# This one service will manage all the different sessions
session_service = InMemorySessionService()
my_user_id = "abajpai"


def get_openai_llm():
    openai_llm = LiteLlm(model="openai/gpt-4o")
    return openai_llm


def get_live_weather_forecast(location: str) -> dict:
    """Gets the current, real-time weather forecast for a specified location in the US.

    Args:
        location: The city name, e.g., "San Francisco".

    Returns:
        A dictionary containing the temperature and a detailed forecast.
    """
    print(f"🛠️ TOOL CALLED: get_live_weather_forecast(location='{location}')")

    #dummy coordinates for "lake tahoe"
    coords_str = "39.0968,-120.0324"
    
    try:
        # NWS API requires 2 steps: 1. Get the forecast URL from the coordinates.
        points_url = f"https://api.weather.gov/points/{coords_str}"
        headers = {"User-Agent": "ADK Example Notebook"}
        points_response = requests.get(points_url, headers=headers)
        points_response.raise_for_status()  # Raise an exception for bad status codes
        forecast_url = points_response.json()["properties"]["forecast"]

        # 2. Get the actual forecast from the URL.
        forecast_response = requests.get(forecast_url, headers=headers)
        forecast_response.raise_for_status()

        # Extract the relevant forecast details
        current_period = forecast_response.json()["properties"]["periods"][0]
        return {
            "status": "success",
            "temperature": f"{current_period['temperature']}°{current_period['temperatureUnit']}",
            "forecast": current_period["detailedForecast"],
        }
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"API request failed: {e}"}


# ADK's built-in `google_search` is a Gemini-side tool and cannot be used with an
# OpenAI model. The *-search-preview models do web search server-side at OpenAI
# instead, so no search tool is declared here(gpt-4o-search-preview rejects function calling) and only OPENAI_API_KEY is needed.
# drop_params lets LiteLLM discard params these models reject (e.g. temperature).
google_search_agent = Agent(
    name="web_searcher",
    model=LiteLlm(model="openai/gpt-4o-search-preview", drop_params=True),
    description="An expert at searching the web for relevant, up-to-date information.",
    instruction="""Accept a search query, search the web, and return a concise
    summary with source URLs.""",
    # no tools — search happens server-side at OpenAI
)

weather_aware_planner_agent = Agent(
    name="weather_aware_planner",
    model=get_openai_llm(),
    description="A trip planner that checks the real-time weather before making suggestions",
    instruction="You are a cautious trip planner. Before suggesting any outdoor activities, you MUST use the `get_live_weather_forecast` tool to check conditions. Incorporate the live weather details into your recommendation.",
    tools=[get_live_weather_forecast, AgentTool(google_search_agent)],
)

print(
    f"🌦️ Agent '{weather_aware_planner_agent.name}' is created and can now call a live weather API!"
)


async def run_weather_aware_planner_agent():

    weather_session = await session_service.create_session(
        app_name=weather_aware_planner_agent.name, user_id=my_user_id
    )
    query = "I want to go hiking near Lake Tahoe, what's the weather like?"
    print(f"🗣️ User Query: '{query}'")
    await run_agent_query(
        weather_aware_planner_agent, query, weather_session, my_user_id
    )


async def run_agent_query(
    agent: Agent, query: str, session: Session, user_id: str, is_router: bool = False
):
    """Initializes a runner and executes a query for a given agent and session."""
    print(f"\n🚀 Running query for agent: '{agent.name}' in session: '{session.id}'...")

    runner = Runner(agent=agent, session_service=session_service, app_name=agent.name)

    final_response = ""
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user"),
        ):
            if not is_router:
                # Let's see what the agent is thinking!
                print(f"EVENT: {event}")
            if (
                event.is_final_response()
                and event.content
                and event.content.parts
                and event.content.parts[0].text
            ):
                final_response = event.content.parts[0].text
    except Exception as e:
        final_response = f"An error occurred: {e}"

    if not is_router:
        print("\n" + "-" * 50)
        print("✅ Final Response:")
        # display(Markdown(final_response))
        print(final_response)
        print("-" * 50 + "\n")

    return final_response


async def main() -> None:

    await run_weather_aware_planner_agent()


if __name__ == "__main__":
    asyncio.run(main())
