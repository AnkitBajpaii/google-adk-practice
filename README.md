# ADK Practice

Practice agents written while following Google's Agent Development Kit (ADK)
hands-on course. Each folder is a self-contained agent package demonstrating **one
orchestration pattern**, kept as a reference to come back to — not as a library.
Boilerplate is repeated on purpose so any single `agent.py` reads on its own.

Agents run on **OpenAI via LiteLLM**, so `OPENAI_API_KEY` is all you need —
except `research_agent`, which also uses Gemini sub-agents.

## Folder structure

| Folder | Pattern |
| --- | --- |
| [day_trip_agent/](day_trip_agent/) | One LLM agent + a search sub-agent wrapped as `AgentTool`. **Start here.** |
| [multi_day_trip_agent/](multi_day_trip_agent/) | Multi-turn memory — one `Session` reused across turns. |
| [weather_aware_planner_agent/](weather_aware_planner_agent/) | A plain Python function as a tool, calling a live REST API. |
| [workflow_agent/](workflow_agent/) | `SequentialAgent`, `LoopAgent`, `ParallelAgent`, shared state via `output_key`, and two routing styles. |
| [agent_as_tool_demo_agent/](agent_as_tool_demo_agent/) | Nested orchestration — tools that invoke agents and pass data via `tool_context.state`. |
| [research_agent/](research_agent/) | A controller delegating to Wikipedia / arXiv / web specialists, then writing a report to a file. |

Every folder has the same shape:

```text
<agent_name>/
├── __init__.py    # exposes the module to `adk web`
└── agent.py       # the agent, plus a runnable demo in __main__
```

## Quick start

**1. Install** — this is a [uv](https://docs.astral.sh/uv/) project, pinned by `uv.lock`:

```powershell
uv sync
```

**2. Add your keys** — one `.env` in the repo root covers every agent:

```ini
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...   # only needed by research_agent
```

**3. Run it** — either in the ADK dev UI, from the repo root:

```powershell
uv run adk web
```

...or as a script, where each agent runs a canned demo query:

```powershell
uv run python -m day_trip_agent.agent
```

All six agents work both ways. ADK prints a `DeprecationWarning` about
`SequentialAgent` on startup — that's expected, not a problem.
