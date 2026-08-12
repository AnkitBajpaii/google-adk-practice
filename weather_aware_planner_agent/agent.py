"""A plain Python function used as a tool, calling a live REST API.

Uses the US National Weather Service, which is free and needs no API key but
only covers the United States. Coordinates come from OpenStreetMap's Nominatim
geocoder, so the tool actually honours whatever `location` it is given.
"""

import asyncio

import requests
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools.agent_tool import AgentTool
from google.genai.types import Content, Part

load_dotenv()

# --- Initialize our Session Service ---
# This one service will manage all the different sessions
session_service = InMemorySessionService()
my_user_id = "abajpai"

# Both APIs ask for a descriptive User-Agent and will throttle or reject requests
# without one. Always pass an explicit timeout so a stalled call cannot hang the
# agent indefinitely.
HTTP_HEADERS = {"User-Agent": "google-adk-practice (ADK learning project)"}
HTTP_TIMEOUT = 15
GEOCODE_URL = "https://nominatim.openstreetmap.org/search"


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

    try:
        # 1. Turn the place name into coordinates.
        geo_response = requests.get(
            GEOCODE_URL,
            params={"q": location, "format": "json", "limit": 1},
            headers=HTTP_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        geo_response.raise_for_status()
        matches = geo_response.json()
        if not matches:
            return {
                "status": "error",
                "message": f"Could not find coordinates for '{location}'.",
            }
        coords_str = f"{float(matches[0]['lat']):.4f},{float(matches[0]['lon']):.4f}"

        # 2. NWS needs two calls: first resolve the coordinates to a forecast URL.
        points_response = requests.get(
            f"https://api.weather.gov/points/{coords_str}",
            headers=HTTP_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        if points_response.status_code == 404:
            return {
                "status": "error",
                "message": (
                    f"'{location}' is outside the US National Weather Service's "
                    "coverage area, which only includes the United States."
                ),
            }
        points_response.raise_for_status()
        forecast_url = points_response.json()["properties"]["forecast"]

        # 3. Then fetch the actual forecast.
        forecast_response = requests.get(
            forecast_url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT
        )
        forecast_response.raise_for_status()

        current_period = forecast_response.json()["properties"]["periods"][0]
        return {
            "status": "success",
            "location": matches[0].get("display_name", location),
            "temperature": f"{current_period['temperature']}°{current_period['temperatureUnit']}",
            "forecast": current_period["detailedForecast"],
        }
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"API request failed: {e}"}
    except (KeyError, IndexError, ValueError) as e:
        # A shape change or non-JSON body should not crash the whole agent run.
        return {"status": "error", "message": f"Unexpected API response: {e}"}


# ADK's built-in `google_search` is a Gemini-side tool and cannot be used with an
# OpenAI model. The *-search-preview models do web search server-side at OpenAI
# instead, so no search tool is declared here (gpt-4o-search-preview rejects
# function calling) and only OPENAI_API_KEY is needed.
# drop_params lets LiteLLM discard params these models reject (e.g. temperature).
google_search_agent = Agent(
    name="web_searcher",
    model=LiteLlm(model="openai/gpt-4o-search-preview", drop_params=True),
    description="An expert at searching the web for relevant, up-to-date information.",
    instruction="""Accept a search query, search the web, and return a concise
    summary with source URLs.""",
    # no tools — search happens server-side at OpenAI
)

weather_aware_planner = Agent(
    name="weather_aware_planner",
    model=get_openai_llm(),
    description="A trip planner that checks the real-time weather before making suggestions",
    instruction=(
        "You are a cautious trip planner. Before suggesting any outdoor activities, "
        "you MUST use the `get_live_weather_forecast` tool to check conditions, "
        "passing the location the user asked about. Incorporate the live weather "
        "details into your recommendation. If the tool reports an error, tell the "
        "user plainly instead of guessing the weather."
    ),
    tools=[get_live_weather_forecast, AgentTool(google_search_agent)],
)

root_agent = weather_aware_planner  # discovered by `adk web`

print(
    f"🌦️ Agent '{weather_aware_planner.name}' is created and can now call a live weather API!"
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
        print(final_response)
        print("-" * 50 + "\n")

    return final_response


async def run_weather_aware_planner_agent():
    weather_session = await session_service.create_session(
        app_name=weather_aware_planner.name, user_id=my_user_id
    )
    query = "I want to go hiking near Lake Tahoe, what's the weather like?"
    print(f"🗣️ User Query: '{query}'")
    await run_agent_query(weather_aware_planner, query, weather_session, my_user_id)


async def main() -> None:
    await run_weather_aware_planner_agent()


if __name__ == "__main__":
    asyncio.run(main())
