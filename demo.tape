# demo.tape — VHS script (https://github.com/charmbracelet/vhs)
#
# Alternative GIF path for contributors who have the `vhs` CLI installed.
# `scripts/make_demo.py` (via `make demo`) is the canonical generator for
# docs/assets/demo.cast + demo.svg; this produces docs/assets/demo.gif
# instead, from the same ~30-second story.
#
# Requirements:
#   - `vhs` installed: https://github.com/charmbracelet/vhs#installation
#   - `memstrand` installed in the shell `vhs` spawns (`pip install -e .`
#     from the repo root, or `pipx install memstrand`)
#   - bash, python3, and mktemp on PATH
#
# Run it:
#   vhs demo.tape
#
# It records into a fresh temp MEMSTRAND_HOME (via `mktemp -d`, captured
# below) on an unused port, so it never touches a real daemon or memory
# store, and is safe to re-run any time.

Output docs/assets/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1000
Set Height 600
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set Padding 20

# -- setup (hidden): fresh temp home, unused port, no daemon in the way ----------------
Hide
Type `export MEMSTRAND_HOME=$(mktemp -d) MEMSTRAND_PORT=58732 && clear`
Enter
Sleep 500ms
Show

# -- 1: a global preference -------------------------------------------------------------
Type `# Your agent learns a preference (global)`
Enter
Sleep 400ms
Type `memstrand remember "Sam prefers concise answers with code first" --category preference`
Enter
Sleep 2s

# Capture the preference's id for the `edit` step below, without showing the plumbing.
Hide
Type `PREF_ID=$(memstrand list --global --json | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['id'])") && clear`
Enter
Sleep 300ms
Show

# -- 2: a task-scoped fact ----------------------------------------------------------------
Type `# ...and a fact while working on one task`
Enter
Sleep 400ms
Type `memstrand remember "Playwright installed for the agent-testing task" --task agent-testing`
Enter
Sleep 2s

# -- 3: scope isolation — no leak ----------------------------------------------------------
Type `# A different task asks about testing setup — no leak`
Enter
Sleep 400ms
Type `memstrand recall "testing setup" --task disk-cleanup`
Enter
Sleep 2.5s

# -- 4: the right task sees its fact + your globals -----------------------------------------
Type `# The right task sees its fact + your globals`
Enter
Sleep 400ms
Type `memstrand recall "testing setup" --task agent-testing`
Enter
Sleep 2.8s

# -- 5: facts change; edits supersede, they don't delete ------------------------------------
Type `# Facts change; edits supersede — they don't delete`
Enter
Sleep 400ms
Type `memstrand edit $PREF_ID --content "Sam now prefers detailed explanations"`
Enter
Sleep 2s

# -- 6: a ready-made context block for a local LLM -------------------------------------------
Type `# Hand your local LLM a ready-made context block`
Enter
Sleep 400ms
Type `memstrand context "write the test plan" --task agent-testing`
Enter
Sleep 3s
