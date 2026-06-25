# 0002 — Repository hygiene

**Date:** 2026-06-24
**Summary:** A small housekeeping push immediately after the initial release: stop
tracking a runtime tooling artifact that should never have been committed, and
extend `.gitignore` so it cannot come back.

This file exists to honor the repo convention from
[`0001-initial-implementation.md`](0001-initial-implementation.md): **every push
adds a `CHANGES/NNNN-*.md`** explaining what changed and why — even the small ones.

---

## What changed

- **Untracked `.claude/`.** A `.claude/scheduled_tasks.lock` file (a transient lock
  created by editor/agent tooling in the working directory) had been swept into the
  first commit by `git add -A`. It is not part of the project and is removed from
  version control with `git rm --cached`.
- **`.gitignore`**: added `.claude/` under the "editor / OS / tooling noise"
  section so the directory is never committed again.

No source, build, test, demo, or documentation content changed. `sha_tests` still
prints `ALL TESTS PASSED`; all three build systems are unaffected.

## Why it matters (the didactic note)
Committed build/tooling artifacts are exactly what the `.gitignore` shipped in
0001 is meant to prevent — this push closes a gap in that net. The lesson: after
the very first `git add -A`, always review `git ls-files` for anything that is not
real source. Here the tree went from 31 tracked files back to the intended 30.
