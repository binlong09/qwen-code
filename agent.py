#!/usr/bin/env python3
"""Minimal Claude Code clone — v0.1. Single-file agent with 4 tools."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

# ---------- Config ----------
MODEL = "qwen3-coder:30b"
BASE_URL = "http://openai:11434/v1"
API_KEY = "ollama"  # any non-empty string; Ollama ignores it
MAX_ITERATIONS = 25
# WORKING_DIR is the root for all tool paths. Defaults to cwd at import; main()
# may overwrite it from --project-root. Tools read this name fresh on each call,
# so reassignment from main() takes effect immediately.
WORKING_DIR = Path.cwd().resolve()
# YOLO_MODE bypasses all approval prompts. Set from --yolo.
YOLO_MODE = False

READ_FILE_MAX_LINES = 2000
BASH_TIMEOUT = 30
BASH_STREAM_CAP = 5000
SEARCH_MAX_MATCHES = 100
DISPLAY_ARGS_CAP = 200
DISPLAY_RESULT_CAP = 500

console = Console()


# ---------- Path safety ----------
class PathError(ValueError):
    """Raised when a path cannot be resolved inside the working directory.

    The message is preformatted with the resolved path, working directory, and a
    hint, so tools can return str(exc) directly to the model.
    """


def _resolve_path(path: str) -> Path:
    """Resolve `path` to an absolute path inside WORKING_DIR.

    Uses Path.resolve() so symlinks are followed; the resolved real path must be
    a descendant of the resolved working directory or PathError is raised.
    """
    wd = WORKING_DIR  # snapshot — WORKING_DIR is a module global, may change
    p = Path(path)
    try:
        resolved = (p if p.is_absolute() else wd / p).resolve()
    except Exception as e:
        raise PathError(
            f"could not resolve path {path!r}: {type(e).__name__}: {e}\n"
            f"  working directory: {wd}"
        )
    if resolved != wd:
        try:
            resolved.relative_to(wd)
        except ValueError:
            raise PathError(
                f"path resolves outside the working directory: {path!r}\n"
                f"  resolved to: {resolved}\n"
                f"  working directory: {wd}\n"
                f"  hint: pass a path relative to the working directory, "
                f"or an absolute path that lies inside it. Symlinks pointing "
                f"outside are rejected."
            )
    return resolved


def _show_lines(path: Path, start: int, end: int) -> str:
    """Return lines [start, end] (1-indexed, inclusive, clamped) with line-number prefix."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"  (could not read file for context: {type(e).__name__}: {e})"
    lines = text.splitlines()
    s = max(1, start)
    e = min(len(lines), end)
    if s > e:
        return "  (no lines to display)"
    return "\n".join(f"  {i:>4}\t{lines[i - 1]}" for i in range(s, e + 1))


# ---------- Tools ----------
def read_file(path: str) -> str:
    try:
        p = _resolve_path(path)
    except PathError as e:
        return f"Error: {e}"
    if not p.exists():
        return (
            f"Error: file not found: {path}\n"
            f"  resolved to: {p}\n"
            f"  working directory: {WORKING_DIR}\n"
            f"  hint: if the file is elsewhere, pass an absolute path or a path "
            f"relative to the working directory."
        )
    if not p.is_file():
        return (
            f"Error: not a regular file: {path}\n"
            f"  resolved to: {p}\n"
            f"  working directory: {WORKING_DIR}"
        )
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
    try:
        p = _resolve_path(path)
    except PathError as e:
        return f"Error: {e}"
    if p.exists():
        return (
            f"Error: file already exists: {path}\n"
            f"  resolved to: {p}\n"
            f"  working directory: {WORKING_DIR}\n"
            f"  hint: write_file only creates new files. To edit, use replace_in_file."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    if content == "":
        n_lines = 0
    else:
        n_lines = content.count("\n") + (0 if content.endswith("\n") else 1)
    return f"Wrote {path} ({n_lines} lines)"


# ---------- Diff parsing for replace_in_file ----------
SEARCH_FENCE = "<<<<<<< SEARCH"
SEPARATOR_FENCE = "======="
REPLACE_FENCE = ">>>>>>> REPLACE"


def _parse_diff(diff: str) -> list[tuple[str, str]]:
    """Parse a diff containing one or more SEARCH/REPLACE blocks.

    Format (fence lines must appear alone on their own lines, exact spelling):
        <<<<<<< SEARCH
        old text
        =======
        new text
        >>>>>>> REPLACE

    Returns list of (search, replace) pairs in order. Raises ValueError on
    malformed input with a message naming the offending line.
    """
    lines = diff.split("\n")
    blocks: list[tuple[str, str]] = []
    state = "between"  # between | in_search | in_replace
    search_buf: list[str] = []
    replace_buf: list[str] = []

    for i, line in enumerate(lines, 1):
        if state == "between":
            if line.strip() == "":
                continue
            if line == SEARCH_FENCE:
                state = "in_search"
                search_buf, replace_buf = [], []
            else:
                raise ValueError(
                    f"line {i}: expected {SEARCH_FENCE!r} to start a block, got {line!r}"
                )
        elif state == "in_search":
            if line == SEPARATOR_FENCE:
                state = "in_replace"
            elif line in (SEARCH_FENCE, REPLACE_FENCE):
                raise ValueError(
                    f"line {i}: unexpected fence {line!r} inside SEARCH section; "
                    f"did you forget the {SEPARATOR_FENCE!r} divider?"
                )
            else:
                search_buf.append(line)
        else:  # in_replace
            if line == REPLACE_FENCE:
                blocks.append(("\n".join(search_buf), "\n".join(replace_buf)))
                state = "between"
            elif line in (SEARCH_FENCE, SEPARATOR_FENCE):
                raise ValueError(
                    f"line {i}: unexpected fence {line!r} inside REPLACE section"
                )
            else:
                replace_buf.append(line)

    if state != "between":
        raise ValueError(
            f"unterminated block (state {state!r}); expected {REPLACE_FENCE!r} to close"
        )
    if not blocks:
        raise ValueError("no SEARCH/REPLACE blocks found in diff")
    return blocks


_DIFF_FORMAT_EXAMPLE = (
    "  <<<<<<< SEARCH\n"
    "  exact text from the file\n"
    "  =======\n"
    "  replacement text\n"
    "  >>>>>>> REPLACE"
)


def _no_match_error(
    path_str: str, resolved: Path, text: str, block_idx: int, n_blocks: int
) -> str:
    lines = text.splitlines()
    preview = "\n".join(f"  {i + 1:>4}\t{lines[i]}" for i in range(min(20, len(lines))))
    block_note = (
        f" (block {block_idx} of {n_blocks}; "
        f"file on disk unchanged — preceding blocks were applied in memory only)"
        if n_blocks > 1
        else ""
    )
    return (
        f"Error: SEARCH text not found in {path_str}{block_note}.\n"
        f"  resolved to: {resolved}\n"
        f"  current state is {len(lines)} lines; first 20 shown below:\n"
        f"{preview}\n"
        f"  hint: the SEARCH text was not found — check for whitespace, indentation, "
        f"tabs vs spaces, or trailing spaces. Do NOT include the right-aligned "
        f"line-number prefix from read_file output."
    )


def _multi_match_error(
    path_str: str,
    resolved: Path,
    text: str,
    search: str,
    block_idx: int,
    n_blocks: int,
) -> str:
    lines = text.splitlines()
    matches: list[tuple[int, int]] = []
    idx = 0
    while True:
        i = text.find(search, idx)
        if i == -1:
            break
        start = text.count("\n", 0, i) + 1
        end = start + search.count("\n")
        matches.append((start, end))
        idx = i + 1
    match_lines = [s for s, _ in matches]

    def _ctx(s: int, e: int) -> str:
        s0, e0 = max(1, s - 2), min(len(lines), e + 2)
        return "\n".join(f"  {j:>4}\t{lines[j - 1]}" for j in range(s0, e0 + 1))

    ctx_blocks = "\n".join(
        f"  Match {n} (lines {s}-{e}):\n{_ctx(s, e)}" for n, (s, e) in enumerate(matches, 1)
    )
    block_note = f" (block {block_idx} of {n_blocks})" if n_blocks > 1 else ""
    return (
        f"Error: SEARCH text matched {len(matches)} times in {path_str} at lines "
        f"{match_lines}{block_note}.\n"
        f"  resolved to: {resolved}\n"
        f"{ctx_blocks}\n"
        f"  hint: include more surrounding lines in the SEARCH section (above or below "
        f"the target) to uniquely identify which match to change. No part of the file "
        f"was modified."
    )


def replace_in_file(path: str, diff: str) -> str:
    try:
        p = _resolve_path(path)
    except PathError as e:
        return f"Error: {e}"
    if not p.exists():
        return (
            f"Error: file not found: {path}\n"
            f"  resolved to: {p}\n"
            f"  working directory: {WORKING_DIR}\n"
            f"  hint: pass a path relative to the working directory."
        )

    try:
        blocks = _parse_diff(diff)
    except ValueError as e:
        return (
            f"Error: malformed diff: {e}\n"
            f"  expected format (one or more blocks):\n"
            f"{_DIFF_FORMAT_EXAMPLE}\n"
            f"  hint: each fence line must appear alone on its own line with exact spelling."
        )

    text = p.read_text(encoding="utf-8")
    for n, (search, replace) in enumerate(blocks, 1):
        count = text.count(search)
        if count == 0:
            return _no_match_error(path, p, text, n, len(blocks))
        if count > 1:
            return _multi_match_error(path, p, text, search, n, len(blocks))
        text = text.replace(search, replace, 1)

    p.write_text(text, encoding="utf-8")
    n = len(blocks)
    return f"Applied {n} block{'s' if n != 1 else ''} to {path}"


def _prompt_approval(command: str) -> bool:
    """Show the command in a distinct panel and prompt for y/N.

    Returns True if the user typed y/yes. EOF or interrupt is treated as denial.
    YOLO_MODE callers should short-circuit before calling this.
    """
    console.print(
        Panel(
            command,
            title="[bold yellow]approval required[/]",
            border_style="yellow",
            expand=False,
        )
    )
    try:
        ans = console.input(r"[bold yellow]Approve?[/] \[y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    return ans in {"y", "yes"}


def bash(command: str, requires_approval: bool) -> str:
    if requires_approval:
        if YOLO_MODE:
            console.print("[dim yellow]  (requires_approval=true; auto-approved by --yolo)[/]")
        elif not _prompt_approval(command):
            return (
                f"User denied approval. The command was NOT executed.\n"
                f"  command: {command}\n"
                f"  hint: if the user denied, adjust the approach (a safer command, "
                f"or explain why this is needed) before retrying."
            )

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


def search(pattern: str, path: str = ".", case_sensitive: bool = False) -> str:
    if shutil.which("rg") is None:
        return (
            "Error: ripgrep (rg) is not installed.\n"
            "  install: `brew install ripgrep` (macOS) | `apt install ripgrep` "
            "(Debian/Ubuntu) | `pacman -S ripgrep` (Arch)\n"
            "  see https://github.com/BurntSushi/ripgrep#installation"
        )
    search_path = path or "."
    try:
        target = _resolve_path(search_path)
    except PathError as e:
        return f"Error: {e}"
    if not target.exists():
        return (
            f"Error: search path not found: {search_path}\n"
            f"  resolved to: {target}\n"
            f"  working directory: {WORKING_DIR}"
        )

    cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
    if not case_sensitive:
        cmd.append("-i")
    cmd += ["--", pattern, search_path]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT,
            cwd=str(WORKING_DIR),
        )
    except subprocess.TimeoutExpired:
        return f"Error: search timed out after {BASH_TIMEOUT}s for pattern {pattern!r}"

    # rg exits 1 with empty output when there are no matches; that's not an error
    if result.returncode == 1 and not result.stdout:
        return f"No matches for pattern {pattern!r} in {search_path}"
    if result.returncode not in (0, 1):
        return (
            f"Error: rg exited with code {result.returncode}\n"
            f"--- stderr ---\n{result.stderr.strip()}"
        )

    out = result.stdout.rstrip("\n")
    lines = out.split("\n") if out else []
    total = len(lines)
    if total > SEARCH_MAX_MATCHES:
        body = "\n".join(lines[:SEARCH_MAX_MATCHES])
        return (
            f"{body}\n"
            f"[...truncated: showing first {SEARCH_MAX_MATCHES} of {total} matches; "
            f"narrow the pattern or path]"
        )
    return body if (body := "\n".join(lines)) else f"No matches for pattern {pattern!r} in {search_path}"


# ---------- Tool schemas ----------
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the working directory. Returns content with right-aligned "
                "line numbers in a 4-character field followed by a tab, then the line content. "
                "Line numbers are display-only — do NOT include them in replace_in_file SEARCH "
                "text. Capped at 2000 lines."
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
                "use replace_in_file to edit existing files."
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
            "name": "replace_in_file",
            "description": (
                "Apply one or more fenced SEARCH/REPLACE blocks to an existing file. "
                "The diff argument contains blocks in this exact format (fence lines must "
                "appear alone on their own lines, exact spelling):\n"
                "  <<<<<<< SEARCH\n"
                "  exact text from the file\n"
                "  =======\n"
                "  replacement text\n"
                "  >>>>>>> REPLACE\n"
                "Multiple blocks can be stacked in one diff; they apply in order. The SEARCH "
                "text in each block must match the file content character-for-character "
                "including whitespace and indentation, and must appear EXACTLY ONCE in the "
                "file's current state. If any block fails (zero or multiple matches, or the "
                "diff is malformed), the entire operation aborts and the file on disk is "
                "unchanged. Do NOT include the right-aligned line-number prefix from "
                "read_file output in SEARCH text. If a SEARCH text would not be unique, "
                "include more surrounding lines until it is."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the working directory.",
                    },
                    "diff": {
                        "type": "string",
                        "description": (
                            "One or more SEARCH/REPLACE blocks. See the tool description "
                            "for the exact fence format."
                        ),
                    },
                },
                "required": ["path", "diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search for content across files using ripgrep. Use search to find where "
                "something is defined or used — prefer this over reading many files looking "
                "for something specific. The pattern is a regex (ripgrep / Rust regex "
                "syntax). Path is relative to the working directory; defaults to '.'. "
                "Case-insensitive by default. Returns matches one per line as "
                "'path:line:content', capped at 100 matches; narrow the pattern if "
                "truncated. Use this before read_file when you don't yet know which file "
                "contains what you need."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern (ripgrep / Rust regex syntax).",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search; relative to the working directory. Defaults to '.'.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "If true, match case exactly. Defaults to false.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command from the working directory. 30-second timeout. "
                "Runs in a FRESH subprocess each call — state does NOT persist between "
                "calls: `cd`, environment variable changes (`export VAR=...`), shell "
                "variable assignments, and background processes all reset before the next "
                "tool call. To run in a subdirectory, prepend `cd <path> && ` to the "
                "command on every call. Returns exit code, stdout, and stderr — each "
                "stream is captured and truncated to 5000 chars. Prefer `rg` (ripgrep) "
                "for code search.\n\n"
                "You MUST set requires_approval: true for commands that modify state "
                "outside the working directory, install software, delete files, make "
                "network requests, or could be destructive in any way. Set false for "
                "read-only commands like `ls`, `cat`, `grep`, `pwd`, `python script.py` "
                "(when the script is known-safe), and most test runs. When in doubt, true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "requires_approval": {
                        "type": "boolean",
                        "description": (
                            "true if the command should require user approval before "
                            "running (destructive, network, install, deletes, etc.); "
                            "false for read-only operations and known-safe scripts."
                        ),
                    },
                },
                "required": ["command", "requires_approval"],
            },
        },
    },
]

DISPATCH: dict[str, Callable[..., str]] = {
    "read_file": read_file,
    "write_file": write_file,
    "replace_in_file": replace_in_file,
    "search": search,
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
_BASE_SYSTEM_PROMPT = """You are a focused coding agent operating in a single working \
directory via five tools: read_file, write_file, replace_in_file, search, bash.

Rules:
- To locate content across files, use the `search` tool (regex via ripgrep). Prefer it
  over reading many files when you don't yet know where something lives.
- Always read a file before editing it. SEARCH text in replace_in_file blocks must match
  the file's content byte-for-byte, including whitespace and indentation.
- The right-aligned line-number prefix shown by read_file is display-only. Never include
  it in replace_in_file SEARCH text.
- Prefer replace_in_file over rewriting a file. Use write_file only for brand-new files.
  replace_in_file accepts multiple SEARCH/REPLACE blocks in one call — group related edits.
- After modifying code, verify it with bash (run the script or its tests).
- For each bash call, set requires_approval=true for destructive, install, network, or
  out-of-tree commands; false for read-only commands and known-safe scripts/tests.
- If a tool returns an error, read it carefully and adjust — do not repeat the same call.
- When replace_in_file reports multiple matches, add more surrounding lines to the SEARCH
  section until it is unique.
- Be terse in your final response. State what you did, which file you changed, and why.
- Never invent file contents. If you need to know what's in a file, read it.
- Stop iterating once the task is verifiably done. If there is no bug or no change needed,
  say so explicitly rather than fabricating one.
"""


def _build_system_prompt() -> str:
    """Prepend the working-directory invariants paragraph to the base prompt.

    The paragraph names the actual resolved working directory so the model can
    reason about path resolution and the fresh-subprocess nature of bash.
    """
    wd_intro = (
        f"Your working directory is {WORKING_DIR}. "
        f"The read_file, write_file, and replace_in_file tools resolve paths "
        f"relative to this directory (or accept absolute paths within it). "
        f"The bash tool also runs from this directory in a fresh subprocess each "
        f"call — `cd` inside a bash command will NOT persist to the next tool "
        f"call. To operate in a subdirectory, either pass relative paths to file "
        f"tools, or prepend `cd <subdir> && ` to every bash command. Do not "
        f"assume state carries between bash calls.\n\n"
    )
    return wd_intro + _BASE_SYSTEM_PROMPT


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
        {"role": "system", "content": _build_system_prompt()}
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
def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-code",
        description="Minimal Claude Code clone — local Qwen via Ollama.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Working directory for all tools (absolute or relative to launch cwd). "
             "Defaults to the current working directory.",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Auto-approve every bash command, even when the model marks "
             "requires_approval=true. Off by default. USE WITH CAUTION.",
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="Task description. If omitted, drops into an interactive REPL.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])

    global WORKING_DIR, YOLO_MODE
    if args.project_root is not None:
        root = Path(args.project_root).expanduser()
        try:
            resolved = root.resolve(strict=True)
        except FileNotFoundError:
            console.print(f"[red]--project-root does not exist:[/] {root}")
            sys.exit(2)
        if not resolved.is_dir():
            console.print(f"[red]--project-root is not a directory:[/] {resolved}")
            sys.exit(2)
        WORKING_DIR = resolved

    YOLO_MODE = bool(args.yolo)
    if YOLO_MODE:
        console.print(
            "[bold red]WARNING:[/] --yolo is set. All bash commands will be "
            "auto-approved, including ones the model marks as destructive."
        )

    console.print(f"[dim]model:[/] {MODEL}  [dim]working dir:[/] {WORKING_DIR}")

    if args.task:
        task = " ".join(args.task)
        console.print(f"[bold]Task:[/] {task}\n")
        run_agent(task)
        return

    console.print("[bold]qwen-code[/] interactive. Type 'exit' or Ctrl-D to quit.")
    history: list[dict] = [{"role": "system", "content": _build_system_prompt()}]
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
