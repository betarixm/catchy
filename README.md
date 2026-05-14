<div align="center">

# 🪤<br>Catchy

**Ca-ca-catch my flag, baby.**

Autonomous AI agent runner for capture-the-flag challenges.

<sub>[Web](https://catchy.bxta.kr)</sub>

<br/>
<br/>

<img src="assets/web.png" alt="Catchy web" width="900" />

</div>

## What is this

Catchy plugs an agent into a CTF challenge, runs it inside a sandboxed workspace, and streams reasoning steps, commands, file changes, costs, and run state into a Django web UI. Each challenge can have multiple threads, and each thread gets its own source extraction, writable workspace, metadata directory, agent configuration, model, credential, and event log.

## Quick start

```bash
# 1. Install dependencies — uv handles the workspace + venv
uv sync

# 2. Set your OpenAI API key, or add it later as a web credential
export OPENAI_API_KEY=sk-...

# 3. Prepare the Django database
uv run python -m catchy.web.manage migrate
uv run python -m catchy.web.manage createsuperuser

# 4. Start the web app
uv run python -m catchy.web.manage runserver
```

Open <http://127.0.0.1:8000>, sign in, then create a credential, model, agent configuration, CTF, and challenge. Start a thread from the challenge page to run the agent and stream its output.

> **Requires** Python 3.14+, [`uv`](https://docs.astral.sh/uv/), and a running Docker daemon.

## Web setup

The web UI stores reusable configuration in the database:

- **Credentials** hold provider API keys. Agent YAML can reference them with OmegaConf interpolation such as `${credential:openai}`.
- **Models** name the model that should be injected into a run.
- **Agents** store YAML like `configurations/codex.yaml` or `configurations/claude-code.yaml`; the `class` field should be a fully qualified import path such as `catchy.codex.CodexAgent` or `catchy.claude_code.ClaudeCodeAgent`.
- **CTFs** group challenges and access rules.
- **Challenges** include a markdown description, optional webhook settings, optional runner config, and a source archive upload or download URL.

## Anatomy of a challenge

Challenges are stored in the web database with an uploaded or downloaded source archive. When a thread starts, Catchy extracts that archive and creates separate source, workspace, and metadata directories for the run.

```text
media/
└── threads/
    └── thread-.../
        ├── source/     # extracted challenge archive
        ├── workspace/  # writable scratchpad mounted into the agent container
        └── metadata/   # run metadata and artifacts kept separate from workspace
```

While a thread is active, use the thread page to queue prompts, steering messages, or a stop request. Public threads can be shared from the web UI.

## Agent Configuration

Agent configurations live in `configurations/*.yaml`. The `class` field is a fully qualified Python import path; Catchy imports it dynamically, validates the YAML with that module's `Configuration` model, then calls `AgentClass.from_configuration(...)`.

```yaml
# configurations/codex.yaml
id: codex-gpt-5.5
class: catchy.codex.CodexAgent
model:
  provider: openai
  name: gpt-5.5
  api_key: ${oc.env:OPENAI_API_KEY}
```

The old shorthand `class: CodexAgent` still resolves to `catchy.codex.CodexAgent`, but new configs should use the full import path.

## Project layout

```text
catchy/
├── packages/
│   ├── core/         # Challenge, Agent, Webhook protocols & models
│   ├── claude-code/  # ClaudeCodeAgent — Claude Code + Docker runtime
│   ├── codex/        # CodexAgent — Codex App Server + Docker runtime
│   └── web/          # Django web UI and thread orchestration
├── configurations/   # Agent YAML configurations
├── challenges/       # Example challenge definitions and source files
└── assets/           # Screenshots and images
```

## Adding a new agent

The `Agent` protocol is minimal: implement `stream(...)`, add a Pydantic-style `Configuration` model in the same module, and expose `from_configuration(...)` on the agent class. `stream(...)` is an async generator: it yields display text and can receive `str | None` steering messages between yields.

```python
from pathlib import Path
from typing import AsyncGenerator

from pydantic import BaseModel

from catchy.core.agent.protocols import Agent
from catchy.core.challenge.models import Challenge
from catchy.core.webhook.models import Webhook

class Configuration(BaseModel):
    id: str

class MyAgent(Agent):
    key = "my-agent"

    def __init__(self, id: str):
        self.id = id

    @staticmethod
    def from_configuration(configuration: Configuration) -> "MyAgent":
        return MyAgent(id=configuration.id)

    async def stream(
        self,
        challenge: Challenge,
        workspace: Path,
        metadata_directory: Path,
        webhook: Webhook | None = None,
    ) -> AsyncGenerator[str, str | None]:
        steering_message = yield "thinking..."
        if steering_message is not None:
            ...
        ...
```

Drop it under `packages/<name>/`, register it in the workspace, then add a YAML file:

```yaml
id: my-agent
class: catchy.my_agent.MyAgent
```

## Roadmap

- [ ] Additional agents (Claude Code, custom)
- [ ] Exportable run transcripts
- [ ] Per-challenge scoreboard
