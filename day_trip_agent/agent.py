import asyncio
from uuid import uuid4
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
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
        3. **Real-Time Focus**: Look up current operating hours, admission prices, and special events for each venue you recommend.
        4. **Mood Matching**: Align suggestions with the requested mood (adventurous, relaxing, artsy, etc.).
        5. **Grounding**: Base hours, prices, and events on what you find on the web, and cite the source URL next to each venue. If you cannot confirm a detail, say so plainly and tell the user to verify it rather than inventing it.

        RETURN itinerary in MARKDOWN FORMAT with clear time blocks and specific venue names.
        """,
    tools=[AgentTool(google_search_agent)],
)

print(f"🧞 Agent '{day_trip_agent.name}' is created and ready for adventure!")


async def run_day_trip_agent():

    session = await session_service.create_session(
        app_name=day_trip_agent.name,
        user_id=my_user_id,
    )

    query = (
        "Plan a relaxing and artsy day trip near Hyderabad, India. Keep it affordable!"
    )
    print(f"User Query : '{query}'")

    await run_agent_query(day_trip_agent, query, session, my_user_id)


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


root_agent = day_trip_agent  # to discover this agent via adk web


async def main() -> None:

    await run_day_trip_agent()


if __name__ == "__main__":
    asyncio.run(main())
