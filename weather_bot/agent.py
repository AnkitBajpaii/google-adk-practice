# @title Import necessary libraries
import asyncio
import logging
import os
import warnings

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Ignore all warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.ERROR)

print("Libraries imported.")

load_dotenv()

print("Loaded all the environment variables")

# --- Verify Keys (Optional Check) ---
print("API Keys Set:")
print(
    f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') else 'No (REPLACE PLACEHOLDER!)'}"
)
print(
    f"OpenAI API Key set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No (REPLACE PLACEHOLDER!)'}"
)
print(
    f"Anthropic API Key set: {'Yes' if os.environ.get('ANTHROPIC_API_KEY') else 'No (REPLACE PLACEHOLDER!)'}"
)

# --- Define Model Constants for easier use ---
MODEL_GEMINI_FLASH = "gemini-flash-latest"
MODEL_GPT_4_1 = "openai/gpt-4.1"
MODEL_CLAUDE_SONNET = "claude-sonnet-4-6"  # You can also try: claude-opus-4-6, etc

print("\nEnvironment configured.")

# Define constants for identifying the interaction context
APP_NAME_GPT = "weather_tutorial_app_gpt"
APP_NAME_CLAUDE = "weather_tutorial_app_claude"

USER_ID_GPT = "user_1_gpt"
USER_ID_CLAUDE = "user_1_claude"

SESSION_ID_GPT = "session_001_gpt"  # Using a fixed ID for simplicity
SESSION_ID_CLAUDE = "session_001_claude"  # Using a fixed ID for simplicity

# --- Tools ---


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city
    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").
    Returns:
         dict: A dictionary containing the weather information.
         Includes a 'status' key ('success' or 'error').
         If 'success', includes a 'report' key with weather details.
         f 'error', includes an 'error_message' key.
    """
    print(f"--- Tool get_weather called for city : {city} ---")
    city_normalized = city.lower().replace(" ", "")

    # Mock weather data
    mock_weather_db = {
        "newyork": {
            "status": "success",
            "report": "The weather in New York is sunny with a temperature of 25°C.",
        },
        "london": {
            "status": "success",
            "report": "It's cloudy in London with a temperature of 15°C.",
        },
        "tokyo": {
            "status": "success",
            "report": "Tokyo is experiencing light rain and a temperature of 18°C.",
        },
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have weather information for '{city}'.",
        }


# --- Agents ---


# -- AGENT using GPT ---
async def run_conversation_using_openai():
    try:
        weather_agent_gpt = Agent(
            name="weather_agent_openai",
            model=LiteLlm(model=MODEL_GPT_4_1),
            description="Provides weather information for specific cities.",
            instruction="""You are a helpful weather assistant.
            When the user asks for the weather in a specific city,
            use the 'get_weather' tool to find the information.
            If the tool returns an error, inform the user politely.
            If the tool is successful, present the weather report clearly.
            """,
            tools=[get_weather],
        )

        print(f"Agent '{weather_agent_gpt.name}' created using model '{MODEL_GPT_4_1}'")

        # InMemorySessionService is simple, non-persistent storage for this tutorial.
        session_service_gpt = InMemorySessionService()

        # Create the specific session where the conversation will happen
        await session_service_gpt.create_session(
            app_name=APP_NAME_GPT, user_id=USER_ID_GPT, session_id=SESSION_ID_GPT
        )

        print(
            f"Session created: App='{APP_NAME_GPT}', User='{USER_ID_GPT}', Session='{SESSION_ID_GPT}'"
        )

        # Create a runner specific to this agent and its session service
        runner_gpt = Runner(
            agent=weather_agent_gpt,
            app_name=APP_NAME_GPT,
            session_service=session_service_gpt,
        )

        print(f"Runner created for agent '{runner_gpt.agent.name}'.")

        # --- Test the GPT Agent ---
        print("\n--- Testing GPT Agent ---")
        # Ensure call_agent_async uses the correct runner, user_id, session_id

        await call_agent_async(
            query="What's the weather in Tokyo?",
            runner=runner_gpt,
            user_id=USER_ID_GPT,
            session_id=SESSION_ID_GPT,
        )

    except Exception as e:
        print(
            f"Could not create or run GPT agent '{MODEL_GPT_4_1}'. Check API Key and model name. Error: {e}"
        )


async def run_conversation_using_claude():
    try:
        weather_agent_claude = Agent(
            name="weather_agent_claude",
            model=LiteLlm(model=MODEL_CLAUDE_SONNET),
            description="Provides weather information for specific cities.",
            instruction="""You are a helpful weather assistant.
            When the user asks for the weather in a specific city,
            use the 'get_weather' tool to find the information.
            If the tool returns an error, inform the user politely.
            If the tool is successful, present the weather report clearly.
            """,
            tools=[get_weather],
        )

        print(f"Agent '{weather_agent_claude.name}' created using model '{MODEL_CLAUDE_SONNET}'")

        # InMemorySessionService is simple, non-persistent storage for this tutorial.
        session_service_claude = InMemorySessionService()

        # Create the specific session where the conversation will happen
        await session_service_claude.create_session(
            app_name=APP_NAME_CLAUDE,
            user_id=USER_ID_CLAUDE,
            session_id=SESSION_ID_CLAUDE,
        )

        print(
            f"Session created: App='{APP_NAME_CLAUDE}', User='{USER_ID_CLAUDE}', Session='{SESSION_ID_CLAUDE}'"
        )

        # Create a runner specific to this agent and its session service
        runner_claude = Runner(
            agent=weather_agent_claude,
            app_name=APP_NAME_CLAUDE,
            session_service=session_service_claude,
        )

        print(f"Runner created for agent '{runner_claude.agent.name}'.")

        # --- Test the GPT Agent ---
        print("\n--- Testing CLAUDE Agent ---")
        # Ensure call_agent_async uses the correct runner, user_id, session_id

        await call_agent_async(
            query="What's the weather in Tokyo?",
            runner=runner_claude,
            user_id=USER_ID_CLAUDE,
            session_id=SESSION_ID_CLAUDE,
        )

    except Exception as e:
        print(
            f"Could not create or run Claude agent '{MODEL_CLAUDE_SONNET}'. Check API Key and model name. Error: {e}"
        )


async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str):
    """Sends a query to the agent and prints the final response."""

    try:
        print(f"\n>>> User Query: {query}")
        print(f"\n🚀 Running query for agent: '{runner.agent.name}' in session: '{session_id}'...")
        # Prepare the user's message in ADK format
        content = types.Content(role="user", parts=[types.Part(text=query)])
        final_response_text = "Agent did not produce a final response."  # Default

        # Key Concept: run_async executes the agent logic and yields Events.
        # We iterate through events to find the final answer.

        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            # You can uncomment the line below to see *all* events during execution
            # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

            # Key Concept: is_final_response() marks the concluding message for the turn.
            if event.is_final_response():
                if event.content and event.content.parts:
                    # Assuming text response in the first part
                    final_response_text = event.content.parts[0].text
                elif (
                    event.actions and event.actions.escalate
                ):  # Handle potential errors/escalations
                    final_response_text = (
                        f"Agent escalated: {event.error_message or 'No specific message.'}"
                    )

        print(f"<<< Agent Response: {final_response_text}")

        return final_response_text
    except Exception as e:
        print(f"Could not run agent. Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(run_conversation_using_openai())
        # asyncio.run(run_conversation_using_claude())
    except Exception as e:
        print(f"An error occurred : {e}")
