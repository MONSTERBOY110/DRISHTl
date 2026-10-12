@AGENTS.md

## Claude Code specifics
- Subhojyoti commits and pushes via GitHub Desktop. **Never run `git commit`, `git push`, `git tag` or any history-changing git command.** Read-only git (`status`, `diff`, `log`) is fine.
- After every task:
  - print the commit message (AGENTS.md §6) with **no `Co-Authored-By:` or other attribution trailer**
  - remind Subhojyoti that the first line goes in GitHub Desktop's **Summary** box and everything else in **Description**
- Python: use the repo venv `venv\Scripts\python.exe` (3.11.9, onnxruntime-gpu; torch is CPU-only for now). Don't use the system Python 3.14.
