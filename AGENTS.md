# AGENTS.md — this repo is guarded by noha

Any AI agent working here follows one rule: **no change is done until `noha prober` shows zero survivors** (untested logic it left behind).

The full integrity loop — gate, probe, repair, verify — lives in [`.noha/INTEGRITY_PROMPT.md`](.noha/INTEGRITY_PROMPT.md). Read it before you touch code, and drive it until the survivor count is 0. A green build must *mean* something.
