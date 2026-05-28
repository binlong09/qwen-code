# qwen-code (v1.1)

A minimal Claude Code clone: a single-file CLI coding agent that uses a local Qwen model
(via Ollama) as the brain, with six tools — `read_file`, `write_file`, `replace_in_file`,
`search`, `bash`, and `task_complete`.

The goal is a working agentic loop in one file (`agent.py`) that can read a small codebase,
locate content with ripgrep, make edits, run tests, ask for approval before risky shell
commands, and explicitly signal completion with verified evidence — autonomously.

## What changed in v1.1

Structured task termination with evidence verification — to push back against the
sycophancy / fabrication failure mode caught by test 4.

- **`task_complete(summary, files_changed, evidence)` tool** — replaces the v1.0
  "model emits text with no tool calls" termination signal. The loop now ends *only*
  when this tool is called and validates. If the model emits a final-looking response
  without calling it, the harness appends a nudge and continues iterating.
- **Required `evidence` field, harness-verified** — every path in `evidence` must have
  been read via `read_file` or returned as a match (or been the search target) of a
  `search` call earlier in this session. Unverified citations fail the call with an
  error naming each one, asking the model to either drop the citation or read the file
  before retrying. Catches both fabricated edits and unverified claims.
- **System-prompt examples for "no change needed"** — explicit examples show that
  `files_changed=[]` is a valid, complete outcome when no edit was required. The model
  is instructed not to invent a bug or pretend to find one.

`SESSION_ACCESSED` (the set of resolved Paths the agent has touched) is populated by
`read_file` on success and by `search` (both the search target and every file that
returned a match — parsed from `rg`'s `path:line:content` output). It persists across
REPL turns alongside conversation history.

## What changed in v1.0

Better edits, model-driven approval, and search.

- **`replace_in_file` replaces `str_replace`** — a more robust edit tool using fenced
  SEARCH/REPLACE blocks (format borrowed from Aider/Cline). Eliminates JSON-string-escape
  failure modes and lets the model batch multiple related edits in a single call.
  Operation is all-or-nothing: if any block fails (zero or multiple matches, or the diff
  is malformed), the file on disk is unchanged.
  ```text
  <<<<<<< SEARCH
  exact text from the file
  =======
  replacement text
  >>>>>>> REPLACE
  ```
- **`requires_approval` on `bash`** — the model itself flags risky commands. The schema's
  required boolean parameter tells the harness whether to prompt the user before running.
  Set true for destructive, install, network, or out-of-tree commands; false for read-only
  ops and known-safe scripts. The harness shows the command in a yellow panel and waits
  for `y/N`. Denial returns a tool result so the model can adjust.
- **`--yolo` CLI flag** — bypasses every approval prompt and auto-approves. Prints a red
  warning at startup. Useful for headless / sandboxed runs; use with caution.
- **`search` tool** — shells out to ripgrep with `--line-number --no-heading --color=never`
  (case-insensitive unless `case_sensitive=true`). Path resolves through `_resolve_path`,
  so search stays inside the working directory. Output is `path:line:content`, capped at
  100 matches with a hint to narrow.

## What changed in v0.1

The harness was made **self-describing** so the model can reason about it correctly:

- **`--project-root` CLI flag** — set the working directory explicitly instead of relying
  on launch cwd.
- **Working-directory paragraph in the system prompt** — the model is told the exact
  absolute path of its working directory, that file tools resolve paths against it, and
  that `bash` runs in a *fresh subprocess each call* so `cd` does not persist.
- **Informative path errors** — `file not found`, `file already exists`, and
  `outside working directory` errors now show the resolved path, the working directory,
  and a hint.
- **Actionable `str_replace` failures** — on zero matches the response includes the first
  20 lines of the file; on multiple matches it includes the line numbers and 2 lines of
  context around each match, with a hint to add more surrounding context.
- **Expanded `bash` and `str_replace` tool descriptions** make the subprocess-isolation
  and display-only-line-numbers invariants explicit in the tool schema.
- **Path-safety hardening** — `_resolve_path` now follows symlinks via `Path.resolve()`
  and verifies the resolved real path is a descendant of the resolved working directory,
  rejecting both `..` traversal and symlinks pointing outside.

## Prereqs

- **Python 3.11+**
- **Ollama** running and reachable at `http://openai:11434/v1`
  (override `BASE_URL` in `agent.py` if your Ollama lives elsewhere, e.g. `http://localhost:11434/v1`)
- The model from the config block in `agent.py` pulled into Ollama:
  ```bash
  ollama pull qwen3-coder:30b
  ```
- **`rg` (ripgrep) on `PATH`** — required for the `search` tool. There is no `grep`
  fallback. Install: `brew install ripgrep` (macOS), `apt install ripgrep` (Debian/Ubuntu),
  `pacman -S ripgrep` (Arch). The tool returns an actionable install hint if `rg` is
  missing.

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

Point the agent at a different directory without `cd`-ing:

```bash
python agent.py --project-root /tmp/qwen-test-1 "Run test_main.py and make it pass."
```

Auto-approve every `bash` command (skip the `y/N` prompt — use with caution):

```bash
python agent.py --yolo "rebuild and run the full test suite"
```

The working directory is the resolved value of `--project-root` (or the launch cwd if the
flag is omitted). File tools cannot escape it; `bash` runs with that as `cwd`, in a fresh
subprocess each call.

## Tools

| Tool              | Purpose                                                                  |
| ----------------- | ------------------------------------------------------------------------ |
| `read_file`       | Read a file (≤ 2000 lines), with display-only line numbers.              |
| `write_file`      | Create a NEW file. Fails if path exists — forces edits via replace_in_file. |
| `replace_in_file` | Apply one or more fenced SEARCH/REPLACE blocks; all-or-nothing.          |
| `search`          | Regex search via ripgrep across files; ≤ 100 matches.                    |
| `bash`            | Run a shell command (30s timeout). Requires the model to flag risky commands via `requires_approval`. |
| `task_complete`   | End the session with a summary, `files_changed`, and harness-verified `evidence`. The only way to terminate the loop. |

## How it works

- The OpenAI Python client points at Ollama's OpenAI-compatible endpoint and uses native
  tool calling.
- Each iteration: stream the model's text to the terminal, collect any tool calls, execute
  them, feed results back as `tool` messages, repeat — up to `MAX_ITERATIONS` (25). The
  loop exits only when `task_complete` is called and its `evidence` validates against
  `SESSION_ACCESSED`. If the model produces text with no tool calls, the harness appends
  a `user` nudge asking it to call `task_complete` and continues.
- Tool calls are printed in cyan with their arguments truncated to 200 chars; tool results
  are previewed at 500 chars in the UI, but the full result is fed back to the model.
- `bash` results are formatted as `exit code: N` + `--- stdout ---` + `--- stderr ---`,
  each stream capped at 5000 chars.
- `read_file` prefixes every line with a right-aligned 4-char line number plus a tab. The
  system prompt tells the model these are display-only and must not appear in
  `replace_in_file` SEARCH text.
- `write_file` refuses to overwrite — forces the agent to use `replace_in_file` for edits.
- `replace_in_file` parses the diff with a strict state machine; SEARCH must occur exactly
  once in the current in-memory state of the file. On 0 matches, the error shows the first
  20 lines; on >1 matches, the line numbers and 2 lines of context around each match. On
  failure the file on disk is untouched, even if earlier blocks already applied in memory.
- `search` shells out to `rg` with `--line-number --no-heading --color=never` and `-i`
  unless `case_sensitive=true`. Results are capped at 100 matches.
- When `bash` is called with `requires_approval=true` and `--yolo` is not set, the harness
  prints the command in a yellow panel and waits for `y/N` on stdin. Denial returns a
  tool result so the model can adjust.
- All paths are resolved through `_resolve_path()` and must stay inside `WORKING_DIR`.
  Symlinks pointing outside are rejected (realpath is checked against the working-dir
  realpath).

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
uses `replace_in_file` to change `a - b` to `a + b`, re-runs, sees `ok`, and reports success.

## Out of scope for v1.1

`list_directory`, repo map / tree-sitter indexing, web fetch, MCP support, subagents,
conversation persistence, multi-file refactor across many edits. v1.2+ territory.
