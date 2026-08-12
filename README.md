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

Not every agent supports both run modes. `adk web` only discovers a folder that
defines a module-level `root_agent`; the script route only works where `agent.py`
has an `if __name__ == "__main__"` block. The last two columns record which is
which, so you don't hit a confusing error picking the wrong one.

| Folder | Pattern demonstrated | `adk web` | script |
| --- | --- | :---: | :---: |
| [day_trip_agent/](day_trip_agent/) | Single LLM agent + a search sub-agent wrapped as `AgentTool`. The simplest end-to-end example — start here. | ✅ | ✅ |
| [multi_day_trip_agent/](multi_day_trip_agent/) | Multi-turn memory: one `Session` reused across three turns so the agent adapts to feedback. | ❌ | ✅ |
| [weather_aware_planner_agent/](weather_aware_planner_agent/) | Plain Python function as a tool, calling a live REST API (US National Weather Service). | ❌ | ✅ |
| [workflow_agent/](workflow_agent/) | `SequentialAgent`, `LoopAgent` (critique/refine with `exit_loop` escalation), `ParallelAgent`, shared state via `output_key` / `{placeholder}`, plus an LLM router dispatching to all of them. | ❌ | ⚠️ see Known issues |
| [agent_as_tool_demo_agent/](agent_as_tool_demo_agent/) | Nested orchestration: async tool functions that invoke `AgentTool.run_async` and pass data between them through `tool_context.state`. | ❌ | ✅ |
| [research_agent/](research_agent/) | Controller delegating to Wikipedia / arXiv / web specialists, then writing the synthesized report to a file. | ✅ | ❌ |

## Setup

This is a [uv](https://docs.astral.sh/uv/) project. Dependencies live in
[pyproject.toml](pyproject.toml) and are pinned by `uv.lock`, so a clone gets the
exact versions this was developed against:

```powershell
uv sync
```

That creates `.venv` and installs everything. Python is pinned to 3.12 by
`.python-version` — the version `uv.lock` was resolved against — and uv downloads
it if you don't have it. `pyproject.toml` accepts 3.12+, but leaving the pin in
place is what keeps a clone identical; ADK 2.6.3 also imports fine on 3.14 if you
bump it.

Then copy [.env.example](.env.example) into each agent folder you want to run:

```powershell
Copy-Item .env.example day_trip_agent\.env
```

```ini
OPENAI_API_KEY=sk-...
# research_agent additionally needs:
GOOGLE_API_KEY=...
```

`.env` files are gitignored — keys stay local.

## Running

**Via the ADK dev UI** (folders marked ✅ in the `adk web` column above):

```powershell
uv run adk web
```

Pick the agent from the dropdown and chat with it.

**As a script** (folders marked ✅ in the `script` column run a canned demo query):

```powershell
uv run python -m day_trip_agent.agent
```

## Known issues

Tracked here rather than silently left for the next person to trip over:

- **`workflow_agent` currently fails at import.** `restaurant_finder_agent` is a
  `model_copy` of `foodie_agent`, which is already a sub-agent of
  `find_and_navigate_agent`. `model_copy` carries `parent_agent` along, so building
  `parallel_research_agent` raises `ValidationError: Agent
  restaurant_finder_agent already has a parent agent`. Building a fresh `Agent`
  for the parallel branch instead of copying fixes it.
- **`research_agent`'s `arxiv_tool` never works.** The module uses `arxiv.Client()`
  but never does `import arxiv`, and the bare `except Exception` swallows the
  resulting `NameError` into a plain error string — so it fails silently and the
  report just omits the papers.
- **`research_agent` never calls `load_dotenv()`.** It imports the function and
  doesn't invoke it. Harmless under `adk web` (ADK loads the agent folder's `.env`
  itself), but it means the module can't be run standalone.
- **`weather_aware_planner_agent` ignores its `location` argument.** The tool
  hardcodes Lake Tahoe coordinates, so asking about any other city still returns
  Tahoe's forecast. The NWS API is US-only regardless.

## Notes

- ADK's built-in `google_search` is a Gemini-side tool and can't be attached to an
  OpenAI model. The workaround used throughout is a `web_searcher` sub-agent on
  `openai/gpt-4o-search-preview`, which searches server-side at OpenAI and takes no
  tools of its own. `drop_params=True` lets LiteLLM discard params that model rejects.
- Passing a plain string like `"openai/gpt-4o"` as `model=` works in ADK 2.6.x — the
  registry resolves `openai/*` to `LiteLlm` automatically. Constructing `LiteLlm(...)`
  explicitly is only needed when passing extra options such as `drop_params`.
- `gemini-3.5-flash` in `research_agent` is ADK 2.6.x's own default model
  (`LlmAgent.DEFAULT_MODEL`), not a typo.
- `.adk/session.db` appears next to an agent after running the dev UI. It is local
  session state and is gitignored.
