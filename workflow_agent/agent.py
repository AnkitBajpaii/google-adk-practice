import re
import os
import sys
import json
import asyncio
import random
import string
from uuid import uuid4
from dotenv import load_dotenv
from google.adk.agents import Agent, LoopAgent, SequentialAgent, ParallelAgent
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
user_id = "abajpai"


def get_openai_llm():
    openai_llm = LiteLlm(model="openai/gpt-4o")
    return openai_llm


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

# --- Agent Definitions for our Specialist Team (Refactored for Sequential Workflow) ---

# ✨ CHANGE 1: We tell foodie_agent to save its output to the shared state.
# Note the new `output_key` and the more specific instruction.
foodie_agent = Agent(
    name="foodie_agent",
    model=get_openai_llm(),
    tools=[AgentTool(google_search_agent)],
    instruction="""You are an expert food critic. Your goal is to find the best restaurant based on a user's request
    When you recommend a place, you must output *only* the name of the establishment and nothing else.
    For example, if the best sushi is at 'Jin Sho', you should output only: Jin Sho
    """,
    output_key="destination",  # ADK will save the agent's final response to state['destination']
)

# ✨ CHANGE 2: We tell transportation_agent to read from the shared state.
# The `{destination}` placeholder is automatically filled by the ADK from the state.
transportation_agent = Agent(
    name="transportation_agent",
    model=get_openai_llm(),
    tools=[AgentTool(google_search_agent)],
    instruction="""You are a navigation assistant. Given a destination, provide clear directions.
    The user wants to go to: {destination}.

    Analyze the user's full original query to find their starting point.
    Then, provide clear directions from that starting point to {destination}.
    """,
)

# ✨ CHANGE 3: Define the SequentialAgent to manage the workflow.
# This agent will run foodie_agent, then transportation_agent, in that exact order.
find_and_navigate_agent = SequentialAgent(
    name="find_and_navigate_agent",
    sub_agents=[foodie_agent, transportation_agent],
    description="A workflow that first finds a location and then provides directions to it.",
)

print("🤖 Agent team assembled with a SequentialAgent workflow!")

# --- Agent Definitions for an Iterative Workflow ---

# A tool to signal that the loop should terminate
COMPLETION_PHRASE = "The plan is feasible and meets all constraints."


def exit_loop(tool_context: ToolContext):
    """Call this function ONLY when the plan is approved, signaling the loop should end."""
    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True
    return {}


# Agent 1: Proposes an initial plan
planner_Agent = Agent(
    name="planner_Agent",
    model=get_openai_llm(),
    tools=[AgentTool(google_search_agent)],
    instruction="You are a trip planner. Based on the user's request, propose a single activity and a single restaurant. Output only the names, like: 'Activity: Exploratorium, Restaurant: La Mar'.",
    output_key="current_plan",
)

# Agent 2 (in loop): Critiques the plan
critic_agent = Agent(
    name="critic_agent",
    model=get_openai_llm(),
    tools=[AgentTool(google_search_agent)],
    instruction=f"""You are a logistics expert. Your job is to critique a travel plan. The user has a strict constraint: total travel time must be short.
    Current Plan: {{current_plan}}
    Use your tools to check the travel time between the two locations.
    IF the travel time is over 45 minutes, provide a critique, like: 'This plan is inefficient. Find a restaurant closer to the activity.'
    ELSE, respond with the exact phrase: '{COMPLETION_PHRASE}'""",
    output_key="criticism",
)

# Agent 3 (in loop): Refines the plan or exits
refiner_agent = Agent(
    name="refiner_agent",
    model=get_openai_llm(),
    tools=[AgentTool(google_search_agent), exit_loop],
    instruction=f"""You are a trip planner, refining a plan based on criticism.
    Original Request: {{session.query}}
    Critique: {{criticism}}
    IF the critique is '{COMPLETION_PHRASE}', you MUST call the 'exit_loop' tool.
    ELSE, generate a NEW plan that addresses the critique. Output only the new plan names, like: 'Activity: de Young Museum, Restaurant: Nopa'.""",
    output_key="current_plan",
)

# ✨ The LoopAgent orchestrates the critique-refine cycle ✨
refinement_loop = LoopAgent(
    name="refinement_loop", sub_agents=[critic_agent, refiner_agent], max_iterations=3
)

# ✨ The SequentialAgent puts it all together
iterative_planner_agent = SequentialAgent(
    name="iterative_planner_agent",
    sub_agents=[planner_Agent, refinement_loop],
    description="A workflow that iteratively plans and refines a trip to meet constraints.",
)

print("🤖 Agent team updated with an iterative LoopAgent workflow!")

# --- Agent Definitions for a Parallel Workflow ---

# Specialist Agent 1
museum_finder_agent = Agent(
    name="museum_finder_agent",
    model=get_openai_llm(),
    tools=[AgentTool(google_search_agent)],
    instruction="You are a museum expert. Find the best museum based on the user's query. Output only the museum's name.",
    output_key="museum_result",
)

# Specialist Agent 2
concert_finder_agent = Agent(
    name="concert_finder_agent",
    model=get_openai_llm(),
    tools=[AgentTool(google_search_agent)],
    instruction="You are an events guide. Find a concert based on the user's query. Output only the concert name and artist.",
    output_key="concert_result",
)

# We can reuse our foodie_agent for the third parallel task!
# Just need to give it a new output_key for this workflow.
restaurant_finder_agent = foodie_agent.model_copy(
    update={"output_key": "restaurant_result", "name": "restaurant_finder_agent"}
)

# ✨ The ParallelAgent runs all three specialists at once ✨

parallel_research_agent = ParallelAgent(
    name="parallel_research_agent",
    sub_agents=[museum_finder_agent, concert_finder_agent, restaurant_finder_agent],
)

# Agent to synthesize the parallel results
synthesis_agent = Agent(
    name="synthesis_agent",
    model=get_openai_llm(),
    instruction=f"""You are a helpful assistant. Combine the following research results into a clear, bulleted list for the user.
    - Museum: {{museum_result}}
    - Concert: {{concert_result}}
    - Restaurant: {{restaurant_result}}
    """,
)

# ✨ The SequentialAgent runs the parallel search, then the synthesis ✨
parallel_planner_agent = SequentialAgent(
    name="parallel_planner_agent",
    sub_agents=[parallel_research_agent, synthesis_agent],
    description="A workflow that finds multiple things in parallel and then summarizes the results.",
)

# For simple weekend guide
weekend_guide_agent = Agent(
    name="weekend_guide_agent",
    model=get_openai_llm(),
    tools=[AgentTool(google_search_agent)],
    instruction="You are a local events guide. Your task is to find interesting events, concerts, festivals, and activities happening on a specific weekend.",
)

# For simple day trip planner
day_trip_agent = Agent(
    name="day_trip_agent",
    model=get_openai_llm(),
    description="Agent specialized in generating spontaneous full-day itineraries based on mood, interests, and budget.",
    instruction="""
    You are the "Spontaneous Day Trip" Generator 🚗 - a specialized AI assistant that creates engaging full-day itineraries.

    Your Mission:
    Transform a simple mood or interest into a complete day-trip adventure with real-time details, while respecting a budget.

    Guidelines:
    1. **Budget-Aware**: Pay close attention to budget hints like 'cheap', 'affordable', or 'splurge'. Use Google Search to find activities (free museums, parks, paid attractions) that match the user's budget.
    2. **Full-Day Structure**: Create morning, afternoon, and evening activities.
    3. **Real-Time Focus**: Search for current operating hours and special events.
    4. **Mood Matching**: Align suggestions with the requested mood (adventurous, relaxing, artsy, etc.).

    RETURN itinerary in MARKDOWN FORMAT with clear time blocks and specific venue names.
    """,
    tools=[AgentTool(google_search_agent)],
)

# We update the router to know about our new, powerful SequentialAgent.
router_agent = Agent(
    name="router_agent",
    model=get_openai_llm(),
    instruction="""
    You are a request router. Your job is to analyze a user's query and decide which of the following agents or workflows is best suited to handle it.
    Do not answer the query yourself, only return the name of the most appropriate choice.

    Available Options:
    - 'foodie_agent': For queries *only* about food, restaurants, or eating.
    - 'weekend_guide_agent': For queries about events, concerts, or activities happening on a specific timeframe like a weekend.
    - 'day_trip_agent': A general planner for any other day trip requests.
    - 'find_and_navigate_agent': For queries that ask to *first find a place* and *then get directions* to it.
    - 'iterative_planner_agent': For planning a trip with a specific constraint that needs checking, like travel time.
    - 'parallel_planner_agent': For queries that ask to find multiple, independent things at once (e.g., a museum AND a concert AND a restaurant).

    Only return the single, most appropriate option's name and nothing else.
    """,
)

# We create a dictionary of all our executable agents for easy lookup.
# This now includes our powerful new workflow agent!
worker_agents = {
    "day_trip_agent": day_trip_agent,
    "weekend_guide_agent": weekend_guide_agent,
    "foodie_agent": foodie_agent,  # For simple food queries
    "find_and_navigate_agent": find_and_navigate_agent,  # sequential agent
    "iterative_planner_agent": iterative_planner_agent,  # loop agent
    "parallel_planner_agent": parallel_planner_agent,  # parallel agent
}


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
                print(f"Event: {event}")

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


async def run_fully_loaded_app():
    queries = [
        # Test Case 1: Simple Sequential Flow
        "Find me the best sushi in Palo Alto and then tell me how to get there from the Caltrain station.",
        # Test Case 2: Iterative Loop Flow
        "Plan me a day in San Francisco with a museum and a nice dinner, but make sure the travel time between them is very short.",
        # Test Case 3: Parallel Flow
        "Help me plan a trip to SF. I need one museum, one concert, and one great restaurant.",
    ]

    for query in queries:
        print(f"\n{'='*60}\n🗣️ Processing New Query: '{query}'\n{'='*60}")
        # 1. Ask the Router Agent to choose the right agent or workflow
        router_session = await session_service.create_session(
            app_name=router_agent.name, user_id=user_id
        )
        print("🧠 Asking the router agent to make a decision...")
        chosen_route = await run_agent_query(
            router_agent, query, router_session, user_id, is_router=True
        )

        chosen_route = chosen_route.strip().strip("`'\". \n")

        print(f"🚦 Router has selected route: '{chosen_route}'")

        # 2. Execute the chosen route
        if chosen_route in worker_agents:
            worker_agent = worker_agents[chosen_route]
            print(f"--- Handing off to {worker_agent.name} ---")
            worker_session = await session_service.create_session(
                app_name=worker_agent.name, user_id=user_id
            )
            await run_agent_query(worker_agent, query, worker_session, user_id)
            print(f"--- {worker_agent.name} Complete ---")
        else:
            print(f"🚨 Error: Router chose an unknown route: '{chosen_route}'")


async def main() -> None:

    await run_fully_loaded_app()


if __name__ == "__main__":
    asyncio.run(main())
