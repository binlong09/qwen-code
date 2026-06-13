# ---- output + look ----
Output coding-agent-demo.gif       # the file VHS will produce
Output coding-agent-demo.mp4       # also render an mp4 (smaller, crisper for web)

Set Shell "bash"
Set FontSize 18
Set Width 1000
Set Height 750
Set Padding 24
Set Theme "Catppuccin Mocha"      # run `vhs themes` to see all options
Set TypingSpeed 60ms              # how fast typing appears

# ---- neutral prompt (hides your username/hostname) ----
Hide
Type "export PS1='$ '" Enter
Type "clear" Enter
Show

# ---- set up the buggy file (kept visible so the demo is self-contained) ----
Type "cat > calc.py << 'EOF'" Enter
Type "def add(a, b):" Enter
Type "    return a - b  # bug" Enter
Type "" Enter
Type "def test_add():" Enter
Type "    assert add(2, 3) == 5" Enter
Type "" Enter
Type 'if __name__ == "__main__":' Enter
Type "    test_add()" Enter
Type '    print("ok")' Enter
Type "EOF" Enter
Sleep 1s

# ---- run the agent ----
Type 'python agent.py "run calc.py, if it fails fix the bug and re-run until it passes"'
Sleep 800ms
Enter

# ---- let the agent work; give it time to think + act ----
Sleep 30s

# ---- linger on the final result ----
Sleep 3s