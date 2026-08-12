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
from google.adk.tools import google_search, ToolContext
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
from google.genai.types import Content, Part
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.google_search_tool import GoogleSearchTool

load_dotenv()

session_service = InMemorySessionService()
my_user_id = "abajpai"

# Assume 'db_agent' is a pre-defined NL2SQL Agent
# For this example, we'll create placeholder agents.

def get_openai_llm():
    openai_llm = LiteLlm(model="openai/gpt-4o")
    return openai_llm

db_agent = Agent(
    name="db_agent",
    model=get_openai_llm(),
    instruction="You are a database agent. When asked for data, return this mock JSON object: {'status': 'success', 'data': [{'name': 'The Grand Hotel', 'rating': 5, 'reviews': 450}, {'name': 'Seaside Inn', 'rating': 4, 'reviews': 620}]}",
)

# --- 1. Define the Specialist Agents ---
# The Food Critic remains the deepest specialist
food_critic_agent = Agent(
    name="food_critic_agent",
    model=get_openai_llm(),
    instruction="You are a snobby but brilliant food critic. You ONLY respond with a single, witty restaurant suggestion near the provided location",
)

# The Concierge knows how to use the Food Critic
conceirge_agent = Agent(
    name="concierge_agent",
    model=get_openai_llm(),
    instruction="You are a five-star hotel concierge. If the user asks for a restaurant recommendation, you MUST use the `food_critic_agent` tool. Present the opinion to the user politely.",
    tools=[AgentTool(agent=food_critic_agent)],
)


# --- 2. Define the Tools for the Orchestrator ---
async def call_db_agent(question: str, tool_context: ToolContext):
    """
    Use this tool FIRST to connect to the database and retrieve a list of places, like hotels or landmarks.
    """
    print("--- TOOL CALL: call_db_agent ---")
    agent_tool = AgentTool(agent=db_agent)
    db_agent_output = await agent_tool.run_async(
        args={"request": question}, tool_context=tool_context
    )
    # Store the retrieved data in the context's state
    tool_context.state["retrieved_data"] = db_agent_output

    return db_agent_output


async def call_concierge_agent(question: str, tool_context: ToolContext):
    """
    After getting data with call_db_agent, use this tool to get travel advice, opinions, or recommendations.
    """
    print("--- TOOL CALL: call_concierge_agent ---")

    # Retrieve the data fetched by the previous tool
    input_data = tool_context.state.get("retrieved_data", "No data found.")

    # Formulate a new prompt for the concierge, giving it the data context
    question_with_data = f"""
    Context: The database returned the following data: {input_data}

    User's Request: {question}
    """

    agent_tool = AgentTool(agent=conceirge_agent)
    concierge_output = await agent_tool.run_async(
        args={"request": question_with_data}, tool_context=tool_context
    )

    return concierge_output


# --- 3. Define the Top-Level Orchestrator Agent ---
trip_data_concierge_agent = Agent(
    name="trip_data_concierge",
    model=get_openai_llm(),
    description="Top-level agent that queries a database for travel data, then calls a concierge agent for recommendations.",
    instruction="""
    You are a master travel planner who uses data to make recommendations.

    1.  **ALWAYS start with the `call_db_agent` tool** to fetch a list of places (like hotels) that match the user's criteria.

    2.  After you have the data, **use the `call_concierge_agent` tool** to answer any follow-up questions for recommendations, opinions, or advice related to the data you just found.

    """,
    tools=[call_db_agent, call_concierge_agent],
    )

print(f"✅ Orchestrator Agent '{trip_data_concierge_agent.name}' is defined and ready.")


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


async def test_trip_data_concierge_agent():
    """
    Sets up a session and runs a query against the top-level
    trip_data_concierge_agent.
    """
    
    # Create a new, single-use session for this query
    concierge_session = await session_service.create_session(
        app_name=trip_data_concierge_agent.name, user_id=my_user_id
    )

    # This query is specifically designed to trigger the full two-step process:
    # 1. Get data from the db_agent.
    # 2. Get a recommendation from the concierge_agent based on that data.
    query = "Find the top-rated hotels in San Francisco from the database, then suggest a dinner spot near the one with the most reviews."
    print(f"🗣️ User Query: '{query}'")

    # We call our existing helper function with the top-level orchestrator agent
    await run_agent_query(
        trip_data_concierge_agent, query, concierge_session, my_user_id
    )


async def main() -> None:

    await test_trip_data_concierge_agent()

if __name__ == "__main__":
    asyncio.run(main())
