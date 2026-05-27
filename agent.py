#!/usr/bin/env python3
"""Minimal Claude Code clone — v0. Single-file agent with 4 tools."""

from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from rich.console import Console

# ---------- Config ----------
MODEL = "qwen3-coder:30b"
BASE_URL = "http://openai:11434/v1"
API_KEY = "ollama"  # any non-empty string; Ollama ignores it
MAX_ITERATIONS = 25
WORKING_DIR = Path.cwd().resolve()

READ_FILE_MAX_LINES = 2000
BASH_TIMEOUT = 30
BASH_STREAM_CAP = 5000
DISPLAY_ARGS_CAP = 200
DISPLAY_RESULT_CAP = 500

console = Console()


# ---------- Path safety ----------
def _resolve_path(path: str) -> Path:
    p = Path(path)
    resolved = (p if p.is_absolute() else WORKING_DIR / p).resolve()
    try:
        resolved.relative_to(WORKING_DIR)
    except ValueError:
        raise ValueError(
            f"Path {path!r} resolves outside the working directory ({WORKING_DIR})"
        )
    return resolved


# ---------- Tools ----------
def read_file(path: str) -> str:
    p = _resolve_path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    if not p.is_file():
        return f"Error: not a regular file: {path}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading {path}: {type(e).__name__}: {e}"
    lines = text.splitlines()
    total = len(lines)
    note = ""
    if total > READ_FILE_MAX_LINES:
        lines = lines[:READ_FILE_MAX_LINES]
        note = f"\n[...truncated: showing first {READ_FILE_MAX_LINES} of {total} lines]"
    body = "\n".join(f"{i + 1:>4}\t{line}" for i, line in enumerate(lines))
    return body + note


def write_file(path: str, content: str) -> str:
    p = _resolve_path(path)
    if p.exists():
        return (
            f"Error: file already exists: {path}. "
            f"Use str_replace to edit existing files."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    if content == "":
        n_lines = 0
    else:
        n_lines = content.count("\n") + (0 if content.endswith("\n") else 1)
    return f"Wrote {path} ({n_lines} lines)"


def str_replace(path: str, old_str: str, new_str: str) -> str:
    p = _resolve_path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    text = p.read_text(encoding="utf-8")
    count = text.count(old_str)
    if count == 0:
        return (
            f"Error: old_str not found in {path}. "
            f"Re-read the file and copy text exactly. "
            f"Do NOT include the right-aligned line-number prefix from read_file output."
        )
    if count > 1:
        match_lines: list[int] = []
        idx = 0
        while True:
            i = text.find(old_str, idx)
            if i == -1:
                break
            match_lines.append(text.count("\n", 0, i) + 1)
            idx = i + 1
        return (
            f"Error: old_str matched {count} times in {path} at lines {match_lines}. "
            f"Include more surrounding context to make the match unique."
        )
    p.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
    return f"Replaced 1 occurrence in {path}"


def bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKING_DIR),
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT,
        )
        stdout, stderr, rc = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout) or ""
        stderr = (e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr) or ""
        stderr += f"\n[timed out after {BASH_TIMEOUT}s]"
        rc = -1

    def _cap(s: str) -> str:
        return s if len(s) <= BASH_STREAM_CAP else s[:BASH_STREAM_CAP] + "\n[...truncated]"

    return (
        f"exit code: {rc}\n"
        f"--- stdout ---\n{_cap(stdout)}\n"
        f"--- stderr ---\n{_cap(stderr)}"
    )


# ---------- Tool schemas ----------
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the working directory. Returns content with right-aligned "
                "line numbers in a 4-character field followed by a tab, then the line content. "
                "Line numbers are display-only — do NOT include them in str_replace arguments. "
                "Capped at 2000 lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the working directory."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create a NEW file with the given content. Fails if the path already exists — "
                "use str_replace to edit existing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the working directory."},
                    "content": {"type": "string", "description": "Full file contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": (
                "Exact string replacement in an existing file. `old_str` must appear EXACTLY ONCE. "
                "Do not include the line-number prefix from read_file output. "
                "On multiple matches, the error returns each match's line number — add more context "
                "around `old_str` to disambiguate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the working directory."},
                    "old_str": {"type": "string", "description": "Exact text to find (unique within the file)."},
                    "new_str": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command in the working directory. 30-second timeout. "
                "Returns exit code, stdout, and stderr. Prefer `rg` (ripgrep) for code search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."}
                },
                "required": ["command"],
            },
        },
    },
]

DISPATCH: dict[str, Callable[..., str]] = {
    "read_file": read_file,
    "write_file": write_file,
    "str_replace": str_replace,
    "bash": bash,
}


def execute_tool(name: str, args: dict[str, Any]) -> str:
    fn = DISPATCH.get(name)
    if fn is None:
        return f"Error: unknown tool {name!r}. Available: {list(DISPATCH)}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}"
    except Exception as e:
        return f"Error: {name} raised {type(e).__name__}: {e}"


# ---------- System prompt ----------
SYSTEM_PROMPT = """You are a focused coding agent operating in a single working directory \
via four tools: read_file, write_file, str_replace, bash.

Rules:
- Always read a file before editing it. Match old_str byte-for-byte from the file contents.
- The right-aligned line-number prefix shown by read_file is display-only. Never include it
  in str_replace arguments.
- Prefer str_replace over rewriting a file. Use write_file only for brand-new files.
- For searching, use bash with ripgrep: `rg "pattern"` or `rg -l "pattern" path/`.
- After modifying code, verify it with bash (run the script or its tests).
- If a tool returns an error, read it carefully and adjust — do not repeat the same call.
- When str_replace reports multiple matches, add more surrounding context to old_str until
  it is unique.
- Be terse in your final response. State what you did and the outcome. No filler.
- Never invent file contents. If you need to know what's in a file, read it.
- Stop iterating once the task is verifiably done.
"""


# ---------- UI helpers ----------
def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"... [+{len(s) - n} chars]"


def _print_tool_call(name: str, args: dict | str) -> None:
    args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    console.print(f"[bold cyan]→ {name}[/] [dim]{_trunc(args_str, DISPLAY_ARGS_CAP)}[/]")


def _print_tool_result(result: str) -> None:
    console.print(f"[dim]  {_trunc(result, DISPLAY_RESULT_CAP)}[/]")


# ---------- Agent loop ----------
def run_agent(user_task: str, history: list[dict] | None = None) -> list[dict]:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    messages = history if history is not None else [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.append({"role": "user", "content": user_task})

    for _ in range(MAX_ITERATIONS):
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                stream=True,
            )
        except Exception as e:
            console.print(f"[red]Model call failed:[/] {type(e).__name__}: {e}")
            return messages

        text_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, str]] = {}
        text_started = False

        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if getattr(delta, "content", None):
                    if not text_started:
                        text_started = True
                    console.print(delta.content, end="", style="white")
                    text_parts.append(delta.content)
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        slot = tool_calls_acc.setdefault(
                            tc.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
        except Exception as e:
            console.print(f"\n[red]Streaming error:[/] {type(e).__name__}: {e}")
            return messages

        if text_started:
            console.print()  # newline after streamed text

        tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
        full_text = "".join(text_parts)

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": full_text or None}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"] or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"] or "{}",
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            return messages

        for i, tc in enumerate(tool_calls):
            name = tc["name"]
            raw_args = tc["arguments"] or "{}"
            call_id = tc["id"] or f"call_{i}"
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    raise ValueError("arguments must decode to an object")
                _print_tool_call(name, args)
                result = execute_tool(name, args)
            except (json.JSONDecodeError, ValueError) as e:
                _print_tool_call(name, raw_args)
                result = f"Error: could not parse arguments ({e}). Raw: {raw_args!r}"
            _print_tool_result(result)
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": result}
            )

    console.print(f"[red]Max iterations ({MAX_ITERATIONS}) reached without completion.[/]")
    return messages


# ---------- CLI ----------
def main() -> None:
    args = sys.argv[1:]
    console.print(f"[dim]model:[/] {MODEL}  [dim]cwd:[/] {WORKING_DIR}")

    if args:
        task = " ".join(args)
        console.print(f"[bold]Task:[/] {task}\n")
        run_agent(task)
        return

    console.print("[bold]qwen-code[/] interactive. Type 'exit' or Ctrl-D to quit.")
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            line = console.input("[bold green]> [/]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if line.strip().lower() in {"exit", "quit"}:
            break
        if not line.strip():
            continue
        history = run_agent(line, history)


if __name__ == "__main__":
    main()
