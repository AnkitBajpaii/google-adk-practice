import os
import sys
import json
import asyncio
import random
import string
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
from google.adk.tools.google_search_tool import GoogleSearchTool

load_dotenv()

session_service = InMemorySessionService()
user_id="abajpai"

def get_openai_llm():
    openai_llm = LiteLlm(model="openai/gpt-4o")
    return openai_llm

# --- Agent Definition: The Adaptive Planner ---

# google_search_agent = Agent(
#     name='web_searcher',
#     model='gemini-3.5-flash',
#     description='An expert at searching the web to find relevant, up-to-date information on a topic using Google Search.',
#     instruction="""Your only task is to accept a search query, use the `GoogleSearchTool` to find information about it, and return a concise summary of the search results.""",
#     tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
# )

# gpt-4o-search-preview rejects function calling
google_search_agent = Agent(
    name='web_searcher',
    model=LiteLlm(model="openai/gpt-4o-search-preview", drop_params=True),
    description='An expert at searching the web for relevant, up-to-date information.',
    instruction="""Accept a search query, search the web, and return a concise
    summary with source URLs.""",
    # no tools — search happens server-side at OpenAI
)

multi_day_agent = Agent(
        name="multi_day_trip_agent",
        model=get_openai_llm(),
        description="Agent that progressively plans a multi-day trip, remembering previous days and adapting to user feedback.",
        instruction="""
        You are the "Adaptive Trip Planner" 🗺️ - an AI assistant that builds multi-day travel itineraries step-by-step.

        Your Defining Feature:
        You have short-term memory. You MUST refer back to our conversation to understand the trip's context, what has already been planned, and the user's preferences. If the user asks for a change, you must adapt the plan while keeping the unchanged parts consistent.

        Your Mission:
        1.  **Initiate**: Start by asking for the destination, trip duration, and interests.
        2.  **Plan Progressively**: Plan ONLY ONE DAY at a time. After presenting a plan, ask for confirmation.
        3.  **Handle Feedback**: If a user dislikes a suggestion (e.g., "I don't like museums"), acknowledge their feedback, and provide a *new, alternative* suggestion for that time slot that still fits the overall theme.
        4.  **Maintain Context**: For each new day, ensure the activities are unique and build logically on the previous days. Do not suggest the same things repeatedly.
        5.  **Final Output**: Return each day's itinerary in MARKDOWN format.
        """,
        tools=[AgentTool(google_search_agent)]
        )

print(f"🗺️ Agent '{multi_day_agent.name}' is created and ready to plan and adapt!")

async def run_agent_query(agent: Agent, query: str, session: Session, user_id: str, is_router: bool = False):
    """Initializes a runner and executes a query for a given agent and session."""
    print(f"\n🚀 Running query for agent: '{agent.name}' in session: '{session.id}'...")

    runner = Runner(agent=agent, session_service=session_service, app_name=agent.name)

    final_response = ""
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user")
        ):
            if not is_router:
                # Let's see what the agent is thinking!
                print(f"Event: {event}")

                if( event.is_final_response()
                   and event.content
                   and event.content.parts
                   and event.content.parts[0].text
                   ):
                    final_response=event.content.parts[0].text
    except Exception as e:
        final_response = f"An error occurred: {e}"

    if not is_router:
            print("\n" + "-" * 50)
            print("✅ Final Response:")
            # display(Markdown(final_response))
            print(final_response)
            print("-" * 50 + "\n")

    return final_response

async def test_multi_day_trip_agent():
    # Create ONE session that we will reuse for the whole conversation
    trip_session = await session_service.create_session(
        app_name=multi_day_agent.name,
        user_id=user_id
        )
    print(f"Created a single session for our trip: {trip_session.id}")

    # --- Turn 1: The user initiates the trip ---
    query1 = "Hi! I want to plan a 2-day trip to Lisbon, Portugal. I'm interested in historic sites and great local food."
    print(f"\n🗣️ User (Turn 1): '{query1}'")

    await run_agent_query(multi_day_agent, query1, trip_session, user_id)

    # --- Turn 2: The user gives FEEDBACK and asks for a CHANGE ---
    # We use the EXACT SAME `trip_session` object!
    query2 = "That sounds pretty good, but I'm not a huge fan of Tower. Can you replace the morning activity for Day 1 with something else historical?"
    print(f"\n🗣️ User (Turn 2 - Feedback): '{query2}'")
    await run_agent_query(multi_day_agent, query2, trip_session, user_id)

    # --- Turn 3: The user confirms and asks to continue ---
    query3 = "Yes, the new plan for Day 1 is perfect! Please plan Day 2 now, keeping the food theme in mind."
    print(f"\n🗣️ User (Turn 3 - Confirmation): '{query3}'")
    await run_agent_query(multi_day_agent, query3, trip_session, user_id)

async def main() -> None:

    await test_multi_day_trip_agent()

if __name__ == "__main__":
    asyncio.run(main())