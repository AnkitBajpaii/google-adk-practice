# ADK Practice — Hello World

Hand-written practice agents built while following Google's Agent Development Kit
(ADK) hands-on course. Each folder is a self-contained ADK agent package kept here
as a **reference for a specific pattern** — not as a library. Files deliberately
repeat boilerplate (`get_openai_llm`, `run_agent_query`) so any one of them can be
read on its own.

Most agents run on **OpenAI via LiteLLM** rather than Gemini, so only an
`OPENAI_API_KEY` is needed. `research_agent` is the exception — it uses Gemini
sub-agents and needs a `GOOGLE_API_KEY`.

## Patterns by folder

| Folder | Pattern demonstrated |
|---|---|
| [day_trip_agent/](day_trip_agent/) | Single LLM agent + a search sub-agent wrapped as `AgentTool`. The simplest end-to-end example — start here. |
| [multi_day_trip_agent/](multi_day_trip_agent/) | Multi-turn memory: one `Session` reused across three turns so the agent adapts to feedback. |
| [weather_aware_planner_agent/](weather_aware_planner_agent/) | Plain Python function as a tool, calling a live REST API (US National Weather Service). |
| [workflow_agent/](workflow_agent/) | `SequentialAgent`, `LoopAgent` (critique/refine with `exit_loop` escalation), `ParallelAgent`, shared state via `output_key` / `{placeholder}`, plus an LLM router dispatching to all of them. |
| [agent_as_tool_demo_agent/](agent_as_tool_demo_agent/) | Nested orchestration: async tool functions that invoke `AgentTool.run_async` and pass data between them through `tool_context.state`. |
| [research_agent/](research_agent/) | Controller delegating to Wikipedia / arXiv / web specialists, then writing the synthesized report to a file. |

## Setup

Python 3.14, using a local virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install google-adk litellm python-dotenv requests wikipedia arxiv
```

Then create a `.env` inside each agent folder you want to run:

```
OPENAI_API_KEY=sk-...
# research_agent additionally needs:
GOOGLE_API_KEY=...
```

`.env` files are gitignored — keys stay local.

## Running

**Via the ADK dev UI** (agents that define `root_agent`):

```powershell
.venv\Scripts\adk.exe web
```

Pick the agent from the dropdown and chat with it.

**As a script** (agents with a `__main__` block run a canned demo query):

```powershell
.venv\Scripts\python.exe -m day_trip_agent.agent
```

## Notes

- ADK's built-in `google_search` is a Gemini-side tool and can't be attached to an
  OpenAI model. The workaround used throughout is a `web_searcher` sub-agent on
  `openai/gpt-4o-search-preview`, which searches server-side at OpenAI and takes no
  tools of its own. `drop_params=True` lets LiteLLM discard params that model rejects.
- Passing a plain string like `"openai/gpt-4o"` as `model=` works in ADK 2.6.x — the
  registry resolves `openai/*` to `LiteLlm` automatically. Constructing `LiteLlm(...)`
  explicitly is only needed when passing extra options such as `drop_params`.
- `.adk/session.db` appears next to an agent after running the dev UI. It is local
  session state and is gitignored.
