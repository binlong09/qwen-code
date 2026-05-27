# qwen-code (v0)

A minimal Claude Code clone: a single-file CLI coding agent that uses a local Qwen model
(via Ollama) as the brain, with four tools — `read_file`, `write_file`, `str_replace`, and
`bash`.

This is **v0**. The goal is a working agentic loop in one file (`agent.py`, ~370 lines) that
can read a small codebase, make an edit, and run a test, autonomously.

## Prereqs

- **Python 3.11+**
- **Ollama** running and reachable at `http://openai:11434/v1`
  (override `BASE_URL` in `agent.py` if your Ollama lives elsewhere, e.g. `http://localhost:11434/v1`)
- The model from the config block in `agent.py` pulled into Ollama:
  ```bash
  ollama pull qwen3-coder:30b
  ```
- `rg` (ripgrep) on PATH if you want the agent to search efficiently

## Install

Using `uv` (recommended):

```bash
cd ~/projects/qwen-code
uv venv
uv pip install -e .
```

Or with plain `pip`:

```bash
cd ~/projects/qwen-code
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Usage

One-shot task (the agent runs until done, then exits):

```bash
python agent.py "find the failing test in this repo and fix it"
```

Interactive REPL (conversation history is kept across turns):

```bash
python agent.py
> read the README
> now add a "Status" section to it
> exit
```

The working directory is whatever you launch from — file tools cannot escape it, and `bash`
runs with that as `cwd`.

## How it works

- The OpenAI Python client points at Ollama's OpenAI-compatible endpoint and uses native
  tool calling.
- Each iteration: stream the model's text to the terminal, collect any tool calls, execute
  them, feed results back as `tool` messages, repeat — up to `MAX_ITERATIONS` (25).
- Tool calls are printed in cyan with their arguments truncated to 200 chars; tool results
  are previewed at 500 chars in the UI but the full result is fed back to the model.
- `bash` results are formatted as `exit code: N` + `--- stdout ---` + `--- stderr ---`,
  each stream capped at 5000 chars.
- `read_file` prefixes every line with a right-aligned 4-char line number plus a tab. The
  system prompt tells the model these are display-only and must not appear in
  `str_replace` arguments.
- `write_file` refuses to overwrite — forces the agent to use `str_replace` for edits.
- `str_replace` requires `old_str` to occur exactly once; on collision it returns each
  match's line number so the agent can disambiguate.
- All paths are resolved through `_resolve_path()` and must stay inside `WORKING_DIR`.

## Config

Edit the top of `agent.py`:

```python
MODEL = "qwen3-coder:30b"
BASE_URL = "http://openai:11434/v1"
MAX_ITERATIONS = 25
```

## Acceptance test

From a shell with Ollama running and `qwen3-coder:30b` pulled:

```bash
mkdir -p /tmp/qwen-code-test
cat > /tmp/qwen-code-test/calc.py <<'EOF'
def add(a, b):
    return a - b  # bug

def test_add():
    assert add(2, 3) == 5

if __name__ == "__main__":
    test_add()
    print("ok")
EOF

cd /tmp/qwen-code-test
python ~/projects/qwen-code/agent.py "run calc.py, if it fails fix the bug and re-run until it passes"
```

Expected behavior: the agent runs `calc.py`, sees the `AssertionError`, reads `calc.py`,
uses `str_replace` to change `a - b` to `a + b`, re-runs, sees `ok`, and reports success.

## Out of scope for v0

Repo map / tree-sitter indexing, web search, MCP, subagents, diff preview/approval UI,
sandboxing, conversation persistence, multi-file transactional edits, planning/todo tools.
Those come in v1+.
