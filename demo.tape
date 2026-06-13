# Primary demo: the agent fixing a bug in a small MULTI-FILE project.
# It locates the bug with `search`, reads the file, edits with replace_in_file,
# verifies with bash, and ends by calling task_complete (evidence-gated).
#
# Renders a raw, real-time mp4; speed.sh post-processes it into the final
# coding-agent-demo.{mp4,gif} (sped up + tight gif). Render with:
#     vhs demo.tape && ./demo/speed.sh
#
# The buggy fixture lives in demo/fixture/ (version-controlled) and is copied
# into a scratch dir so the recording never touches the repo.

Output coding-agent-demo-raw.mp4

Require rg

Set Shell "zsh"
Set FontSize 18
Set Width 1100
Set Height 820
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

# ---- neutral prompt + stage the buggy multi-file fixture off-screen ----
Hide
Type "export PS1='$ '" Enter
Type "rm -rf /tmp/shop-demo && mkdir -p /tmp/shop-demo && cp ~/projects/qwen-code/demo/fixture/*.py /tmp/shop-demo/ && cd /tmp/shop-demo" Enter
Type "clear" Enter
Show

# ---- show the project + where the bug lives ----
Type "ls" Enter
Sleep 1500ms
Type "cat pricing.py" Enter
Sleep 2500ms

# ---- run the agent against this multi-file project ----
Type 'python3 ~/projects/qwen-code/agent.py --project-root /tmp/shop-demo "test_cart.py is failing because the loyalty discount is wrong. Find where the discount is calculated, fix it, and run test_cart.py to confirm it passes."'
Sleep 800ms
Enter

# ---- let the agent work (local 30b ~2 min; headroom so it never clips) ----
Sleep 140s

# ---- linger on the final task-complete panel ----
Sleep 3s
