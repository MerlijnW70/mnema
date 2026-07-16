# AGENTS.md — this repo is guarded by internal-tool

Any AI agent working here follows one rule: **no change is done until `internal-tool prober` shows zero survivors** (untested logic it left behind).

The full integrity loop — gate, probe, repair, verify — lives in [`.internal-tool/INTEGRITY_PROMPT.md`](.internal-tool/INTEGRITY_PROMPT.md). Read it before you touch code, and drive it until the survivor count is 0. A green build must *mean* something.
