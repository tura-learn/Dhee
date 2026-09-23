# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [7.4.0] - 2026-09-23 - Decisions: memory that judges before it stores and before it speaks

Several steps in Dhee were never generation. Is this fact about the person?
Which label does it take? Does it restate something already stored? Which of
these memories would change the next reply? Each was either answered by asking
an LLM and parsing a verdict out of prose, or not asked at all. They are now
answered by a decision model — TypeSafe's Jev, which returns a calibrated
probability per option in well under a second — behind a new, opt-in
`MemoryConfig.decisions` section. Off by default; every failure falls back to
the previous behaviour.

Measured on real student stores before the change: 81 of 186 facts were not
about the student, 2 of 105 facts about the student used a predicate any reader
joined on, and one difficulty was stored eight ways.

- **`dhee.decisions`** — `JevDecider` (OpenRouter's `/api/alpha/decisions` or
  TypeSafe's `/v1/systemone`; never raises, never retries, `DHEE_DECISIONS=off`
  switches it off process-wide) and the `Decider` protocol for any other.
- **The fact gate** (`FactGate`), wired between extraction and storage in the
  write pipeline. With `fact_subjects` set, facts not about the subject are not
  stored; with `fact_vocabulary` set, every fact is filed under one of its
  predicates or not stored; and a fact that restates a stored one is pointed at
  it, so storage reaffirms instead of inserting. Replayed over a real store:
  267 facts became 42 labelled ones, and a run of 39 restatements became 12.
  The facts go in the decision's *state*, not only its questions — with the
  source text alone as state the model judged the text, not the fact.
- **The extractor is told the vocabulary** when one is configured.
- **`select_relevant`** — the read gate: candidate memories in, one Noul per
  candidate ("would knowing this change the next reply?"), only what clears the
  floor out, one slot per idea.
- **`regate_facts`** — puts a store that grew without the gate through it once:
  retires what is not about the subject, relabels, merges restatements into the
  first way a thing was said. Nothing is deleted; `dry_run` writes nothing. A
  real store went from 293 active facts to 22.
- **Distillation no longer re-distils a day on every run.** The sleep cycle
  targets yesterday and runs every few minutes, and nothing recorded which
  episodes had been read: one student store distilled the same 9 episodes 11
  times in a morning into 98 near-identical memories. A `distilled_episodes`
  ledger (plus existing provenance) now marks every episode once its batch is
  processed, and a run with no new episodes is skipped.

## [7.3.1] - 2026-08-26 - Embeddings ask for floats

- **`OpenAIEmbedder` now sends `encoding_format` explicitly, defaulting to
  `float`.** It never sent the field at all, which is not the same as sending
  nothing: recent versions of the OpenAI SDK fill it in themselves with
  `base64` as a payload optimisation. Providers that do not implement base64 —
  OpenRouter's Nvidia embedding models among them — answer `400 Nvidia
  embeddings do not support base64 encoding_format`, once per chunk, forever.
  The failure is caught and logged rather than raised, so the visible symptom
  is not an error but an index that silently never fills: memory reports
  `ready` and contains nothing.

  `float` is what the API documents as its default and what every
  implementation accepts, and the SDK decodes both encodings to the same list
  of floats, so nothing downstream can tell the difference. `embeddings/nvidia.py`
  has always passed it; this is the same fix on the generic client, which is
  what a compatible provider actually uses. Override via
  `config["encoding_format"]` if a provider ever wants otherwise.

## [7.3.0] - 2026-07-29 - Document corpus lane

- **New `dhee.corpus` package: a third lane beside beliefs and world memory.**
  Belief memory answers "what is true about the user and the work"; a corpus
  answers "what does this folder of documents say". They must not share a store:
  `dhee/memory/quality.py` deliberately classifies `doc_chunk` and anything
  carrying a `source_path` as artifact-like rather than belief, so that a
  thousand fragments of a PDF cannot drown a handful of real beliefs. Documents
  now have a place they are supposed to live instead of a filter they have to
  survive. The corpus keeps its own tables and its own vector collection, and is
  never admitted into memory.

- **Folder change tracking shaped like git's object model** (`dhee.corpus.tree`).
  Files are addressed by the sha256 of their bytes, directories are Merkle nodes
  hashed from their sorted children, and a stat cache carries hashes forward for
  files whose size and mtime are both unchanged. The result is that re-indexing
  costs O(changed) rather than O(files): a rename or a move is detected as one
  file rather than a delete plus an add, and so never re-extracts or re-embeds;
  duplicate content across folders is extracted and embedded once; and a re-scan
  of a settled folder does no network work at all. That last property is what
  makes attaching a filesystem watcher safe.

- **Page-aware chunking with citations** (`dhee.corpus.extract`,
  `dhee.corpus.search`). Chunks never straddle a page boundary and carry their
  page through to the answer, because a retrieval hit is only worth as much as
  the citation attached to it. Retrieval is vector recall then cross-encoder
  rerank; a missing or failing reranker degrades to vector order rather than
  failing the search.

- **Extraction is injected, not imported** (`TextExtractor` protocol). Dhee ships
  a plain-text and text-layer-PDF reader so the corpus works standalone; OCR is a
  large bundle that belongs to the host application. Dhee still installs and runs
  on its own.

- **New `openrouter` embedder** (`dhee.embeddings.openrouter`) and
  `OpenRouterReranker`. Two things about that API are easy to lose time on: the
  public `/models` listing returns chat models only, so embedding and reranking
  models look absent until you query `/models/{id}/endpoints`; and `dimensions`
  must not be sent, since the vector width is fixed by the model. Both are
  handled here, with batching and retry-with-backoff.

## [7.2.7] - 2026-07-20 - Kuzu made optional (Python 3.14 install fix)

- Moved `kuzu` out of core dependencies: it ships no wheels for Python 3.14+
  (upstream development has stopped), so `pip install dhee` on a 3.14 machine
  tried to compile the Kùzu C++ engine from source and bricked the install.
  Core capture, checkpoints, embedding recall, and all four MCP tools work
  without it.
- The causal-scene graph projection now degrades cleanly when kuzu is absent:
  `CausalGraphProjection.if_available()` returns `None`, checkpoint sync
  no-ops, and `dhee causal` / `dhee graph` commands raise an actionable
  install hint (`pip install 'dhee[graph]'`, Python <= 3.13) instead of an
  import crash. The graph stays a derived projection — SQLite remains truth,
  and `dhee graph rebuild` reconstructs it after installing the extra.
- `graph` and `all` extras carry a `python_version < '3.14'` marker so no
  install path ever attempts the source build.

## [7.2.6] - 2026-07-11 - Hyper-context skill read fix

- Fixed `AttributeError: 'Skill' object has no attribute 'get'` in
  `buddhi.get_hyper_context`: skill stores may return Skill objects or dicts;
  the context builder now reads both instead of silently degrading the
  relevant-skills section on every call.

## [7.2.5] - 2026-07-03 - Clear-water lifecycle repair

- Updated the default NVIDIA extraction/generation model to
  `moonshotai/kimi-k2.6` and added a live provider model ping in Dhee doctor.
- Enabled scene summarization by default and wired lifecycle enrichment to fill
  missing scene summaries with deterministic fallback summaries.
- Added scene-noise audit and repair for operational/test prompt scenes,
  tombstoning active noise while preserving it as episodic events.
- Expanded `DHEE_DATA_DIR` tildes consistently and hardened pytest data-dir
  isolation so fixture users do not write into the live vault by accident.

## [7.2.4] - 2026-07-02 - Clear-water memory enrichment

- Deferred enrichment now runs structured engram extraction and marks honest
  `failed` and `degraded` states instead of silently completing factless rows.
- Operational tool, screen, and trajectory noise is stored as episodic events
  instead of polluting semantic memory recall.
- Added re-extraction support for complete memories that missed structured
  facts, plus deterministic preference and plan extraction paths.

## [7.2.2] - 2026-06-06 - Split provider memory profiles

- Added provider runtime profiles that let agent LLMs use OpenAI, Anthropic,
  Gemini, Ollama, or NVIDIA while Dhee memory defaults to NVIDIA embeddings and
  reranking.
- Added explicit split-key routing so Chotu/agent API keys do not leak into
  Dhee embedder or reranker configs when the providers differ.
- Added fact-checked provider defaults for NVIDIA Nemotron embedding/reranking,
  OpenAI GPT and embeddings, Anthropic Claude, Gemini models and embeddings,
  and Ollama local defaults.
- Added Anthropic LLM support and provider-profile tests for split memory
  configurations.

## [7.2.1] - 2026-05-26 - Native agent providers

- Added Dhee's universal agent runtime for provider-neutral `before`, event,
  tool, and finish flows.
- Added native provider integrations for ElevenLabs voice agents, Gemini API
  agents, and OpenAI Responses API agents.
- Added the ElevenLabs HTTP sidecar, server-tool profile, post-call webhook
  checkpointing, and voice memory policy.
- Documented the few-line integration path for adding Dhee memory to existing
  agents.

## [7.2.0] - 2026-05-26 - Proactive workspace memory and init polish

- Made Dhee workspace behavior explicitly opt-in: Codex/Claude hooks and router
  enforcement now stay quiet outside folders initialized with `dhee init`.
- Expanded `dhee init` and `dhee link` to support plain folders and git URLs,
  while preserving git hooks for real repositories.
- Added calmer `dhee init` terminal output, typo suggestions for unknown
  commands, and built-in shell completion for bash, zsh, and fish.
- Improved Chotu/Dhee memory retrieval quality: sparse vector results now merge
  DB lexical recall instead of hiding exact durable memories.
- Hardened Narrative Scene Intelligence by rejecting orphan `scene_event`
  writes and making SceneCard summaries lead with story progress and durable
  facts instead of evidence labels.
- Updated installer docs/demo assets and kept PyPI metadata at
  `Development Status :: 5 - Production/Stable`.

## [7.1.0] - 2026-05-22 - World memory layer and context compiler

- Reframed Dhee as the world memory layer and context compiler for AI agents,
  with a polished README that explains the agent workflow, MCP setup,
  retrieval stack, and production boundaries.
- Added SQLite-first Narrative Scene Intelligence v1:
  `Series -> Season -> Episode -> Scene -> SceneCard -> MemoryItem`.
- Added canonical scene intelligence MCP tools in slim and full servers:
  `dhee_scene_start`, `dhee_scene_event`, `dhee_scene_end`,
  `dhee_scene_context`, and `dhee_narrative_prior`.
- Made SceneCards the canonical prompt-safe retrieval object while keeping
  existing temporal scene tools compatible.
- Wired semantic SceneCard retrieval to the NVIDIA/Nemotron embedder and
  reranker path by default when available, with deterministic filters for
  privacy, secrets, proof gates, categories, markers, recency, and outcome.
- Added auditable Episode, Season, and Series rollups with Gemma 4 31B as the
  routine distillation model and Kimi k2.6 as the Series-level strategic model.

## [7.0.2] - 2026-05-21 — Chotu dependency compatibility

- Relaxed Dhee's tree-sitter dependency family from `<0.24` to `<0.26` so
  downstream agents such as Chotu can depend on `tree-sitter>=0.25.0` without a
  resolver conflict.
- Validated Dhee repo cognition against `tree-sitter==0.25.2` with current
  Python, JavaScript, TypeScript, and TSX grammar wheels.
- Updated the curl installer floor to prefer this compatibility release.

## [7.0.1] - 2026-05-21 — Release quality hardening

- Hardened context rollover so context-debt summaries only count the active
  post-rollover task window and expose explicit rollover health.
- Expanded memory-quality quarantine for oddly phrased tool/command failures so
  operational noise does not re-enter personal or project recall.
- Made profile extraction stricter: raw docs, passive evidence, test fixtures,
  and operational events cannot create durable user/project profiles.
- Added immediate persisted-profile cleanup so contaminated profiles are
  sanitized, renamed, or made inert continuously rather than only during manual
  repair.
- Updated the curl installer floor to prefer this hardened 7.0.1 release.

## [7.0.0] - 2026-05-20 — Production-ready Developer Brain

- Marked Dhee as the production-ready local developer brain for coding agents:
  memory, repo cognition, handoff, context routing, task contracts, and proof
  bundles for Codex, Claude Code, Cursor, Cline, Gemini CLI, and MCP clients.
- Reframed the README into a shorter product-first guide with explicit
  boundaries: Dhee is not a model, autocomplete tool, or hosted vector database.
- Replaced the README visuals with three Excalidraw-style diagrams that show
  the agent loop, repo-impact reasoning, and what Dhee does beside a coding
  agent.
- Hardened memory quality as a mandatory contract: canonical personal/project
  facts, passive evidence, test fixtures, and operational events no longer
  compete in the same recall lane.
- Added repair and audit paths for legacy memory pollution, suppressed test
  fixtures, queued/degraded rows, and vector reconciliation.
- Standardized the high-quality semantic-memory defaults around the NVIDIA
  OpenAI-compatible stack, zvec through `dhee-accel`, and the 2048-dimensional
  Nemotron embedding profile.
- Added first-class route and component maps to the persistent repo brain, plus
  impact output for touched files, likely tests, impacted routes/components,
  owners, and failure signatures.
- Installed Kuzu-backed graph support as part of the production repo-cognition
  stack and kept the package metadata at `Development Status :: 5 -
  Production/Stable`.

## [6.2.0] - 2026-05-15 — Production launch hardening

- Marked the public package metadata as `Development Status :: 5 -
  Production/Stable` for the Product Hunt launch.
- Fixed the release wheel so the bundled `engram_bus` handoff package is
  actually included instead of being pruned from the source distribution.
- Hardened `install.sh` with a verified handoff-bus check, a GitHub repair
  fallback if PyPI is stale or broken, `DHEE_INSTALL_PACKAGE` for reproducible
  installer smoke tests, and `DHEE_INIT_REPO` for non-interactive repo wire-up.
- The installer now explicitly points new users at `dhee ui` after install so
  they can open the local command center, folders canvas, and firewall without
  hunting through docs.

## Unreleased — Developer Brain split

- Public Dhee is now positioned and packaged as **Dhee Developer Brain**:
  local memory, handoff, harness setup, and git-backed repo context.
- Rewrote the README as a concise first-read product page focused on why Dhee
  matters, the UI demo, install, integrations, benchmarks, and the
  public-core/paid-team-layer boundary.
- Restored the public `dhee ui` Sankhya workspace app: router screen, infinite
  folders canvas, workspace/task/memory/context views, and the FastAPI bridge
  that maps those screens onto local Dhee state.
- Added product-grade UI screens for Dhee's context-governance workflow:
  Command Center, Context Firewall, Handoff Hub, Proof Replay, Repo Brain,
  Learning Inbox, and Portability & Trust.
- Added `dhee demo token-router`, a deterministic context-firewall demo that
  shows raw tool-output tokens, digest tokens, savings, and expansion pointers
  without requiring a live agent session.
- Added a public `SECURITY.md` with Dhee's local-first trust boundaries,
  threat model, `.dheemem`/repo-context/daemon controls, reporting process, and
  public-core vs paid-governance security split.
- Added canonical `dhee://` URI aliases over DheeFS for stable cross-tool
  references such as `dhee://state/current` and `dhee://handoff/latest`.
- Added `dhee runtime status|restart|stop|doctor` with a local-only runtime
  daemon, managed-venv visibility, and doctor integration.
- `dhee shell`, MCP `dhee_shell`, and compiled context actions now use the
  local runtime daemon when it is healthy, with automatic fallback and
  `DHEE_RUNTIME_DISABLE=1` escape hatch.
- `dhee uninstall` now performs packaging-grade cleanup: stops the daemon,
  disables native harness wiring, removes only installer-owned symlinks, strips
  the exact managed `# dhee` shell PATH block, and deletes the managed data/venv
  directory.
- MCP `dhee_read` and `dhee_grep` now use the local runtime daemon when
  healthy. MCP `dhee_bash` can also use the daemon, but only when the daemon
  process is started with `DHEE_RUNTIME_ENABLE_BASH=1`, a cwd allowlist, and a
  timeout cap; successful results include runtime audit metadata.
- Source-side read routing now extracts richer language-aware digests for
  TS/TSX components and types, Java contracts, shell scripts, SQL objects,
  and log severity signals.
- Router quality reports now include explicit release-facing quality gates
  for token savings, expansion rate, projected cache-read per turn, and
  context-governance incidents.
- Router replay now supports Claude Code and Codex JSONL transcript streams
  plus golden annotations for task parity scores and stale-context incidents.
- Added `dhee router gate` for CI/release gating and wired the checked-in
  Claude/Codex golden replay corpus into GitHub Actions. It exits non-zero on
  failed replay quality gates and supports `--allow-insufficient` for partial
  telemetry jobs.
- Added `dhee router harvest` and `dhee router corpus` to grow the golden
  replay suite from real Claude Code/Codex sessions without checking in raw
  prompts, tool outputs, absolute paths, or secrets. Harvested annotations are
  marked `needs_review` until a human validates task parity.
- Golden replay reports now count `pending_review_sessions`, release gates fail
  when included annotations are still pending, and `dhee router annotate` can
  promote a reviewed session to `pass` or `fail` without hand-editing JSONL.
- Hardened signed `.dheemem` v1 import/inspect validation: manifest signature
  failures now report cleanly, required payload files and `handoff.json` are
  enforced, and duplicate, unexpected, absolute, or traversal archive members
  are rejected before import.
- `.dheemem` import and dry-run results now include a compact
  `handoff_bootstrap` summary from the signed `handoff.json`, so a receiving
  harness can inspect continuity before or after import.
- `.dheemem` packs now also carry signed repo-shared context payloads
  (`repo_context/manifest.json` and `repo_context/entries.jsonl`); import
  dry-runs report the repo-context bootstrap and merge/replace can restore
  entries into a target repo while rejecting tampered, symlinked, or
  likely-secret-bearing context.
- Public Dhee exposes local CLI/MCP/data primitives so dashboard products can
  render governance without duplicating core context logic.
- Added repo-shared context commands: `dhee link`, `dhee unlink`,
  `dhee links`, `dhee promote`, `dhee demote`, and `dhee context`.
- Repo-shared context uses append-only `.dhee/context/entries.jsonl` with
  conflict detection for concurrent developer edits.

## [6.1.0] - 2026-04-24 — Injection cleanup

Three surgical fixes to the `<dhee v="1">` UserPromptSubmit injection so
stale and irrelevant context never leaks into a turn.

- `dhee/db/sqlite_analytics.py` — new `close_stale_shared_tasks()` bulk
  closes any `status='active'` row whose `updated_at` is older than the
  configured window (default 24h). ISO-8601 strings sort lexicographically
  in SQLite, so the comparison is index-friendly with no per-row parsing.
- `dhee/core/shared_tasks.py` — `shared_task_snapshot()` now calls the
  bulk close before resolving, and adds a strict repo filter via
  `_task_matches_repo()` so a task anchored on a sibling repo never
  surfaces in the active workspace's snapshot.
- `dhee/router/edit_ledger.py` — `record()` now stamps `DHEE_SESSION_ID`
  and the recording cwd; `summarise()` filters by session, repo, and a
  6-hour freshness window, and unconditionally drops `/tmp/`,
  `/private/tmp/`, and `/var/folders/` paths. Backward-compat: rows
  missing the new fields lapse out via the freshness window.
- `dhee/hooks/claude_code/__main__.py` — `handle_user_prompt` now gates
  the `<shared>` block on per-turn embedder cosine similarity between
  the prompt and the task title + last result digest. Default threshold
  0.50, override via `DHEE_SHARED_RELEVANCE_THRESHOLD`. Fails closed on
  embedder errors — better to drop than re-emit the noise.
- Tests: `tests/test_oss_61_quickfix.py` covers the three fixes
  end-to-end (auto-close, repo filter, ledger filters, relevance gate).

## [5.1.0] - 2026-04-21 — gstack adapter

Third first-class harness target: `dhee install gstack`. gstack (Garry
Tan's 23-skill Claude Code skill pack) keeps its memory under
`~/.gstack/projects/<slug>/` as siloed JSONL + markdown. This release
adds a read-only adapter that ingests every gstack learning, timeline
event, review, and checkpoint section into Dhee's existing memory
pipeline so gstack users get semantic search, consolidation, correction,
and episodic rehydration without rewriting any gstack code.

- New: `dhee/adapters/gstack.py` + `dhee/adapters/gstack_parser.py` —
  detect, backfill, and delta tail-ingest with per-project cursor
  manifest at `$DHEE_DATA_DIR/gstack_manifest.json`.
- New: `dhee install gstack` (also `dhee install --harness gstack`) and
  `dhee adapters gstack [status|reingest|clear]`.
- `dhee harness status` / `enable` / `disable` now accepts `--harness gstack`.
- Claude Code session hooks call `tail_ingest()` on `SessionStart` and
  `Stop`; no-op unless the user has explicitly run `dhee install gstack`.
- Zero mutation of gstack files. Respects `$GSTACK_HOME`. Runs gstack's
  own prompt-injection denylist before writing so we never ingest what
  gstack itself would reject.
- Docs: `docs/adapters/gstack.md` maps gstack's six memory failure modes
  onto the existing Dhee components that fix each one.
- Tests: `tests/test_gstack_adapter.py` covers backfill, tail,
  idempotency, checkpoint sectioning, uninstall, graceful skip when
  gstack is absent, and injection-safe refusal.

## [5.0.0] - 2026-04-20 — Portable Memory OS Release

Native Claude Code + Codex on one shared Dhee kernel. This release turns Dhee from "memory + router for one harness" into a portable memory OS with native harness install, host-parsed artifact reuse, continuity/handoff, shared-task collaboration, and signed export/import packs.

### Native harness layer

- `dhee install --harness all` configures both Claude Code and Codex against the same `~/.dhee` root.
- Claude Code remains hook-native for routing, memory updates, and context injection.
- Codex is now first-class through native config wiring plus incremental event-stream sync, so post-tool results and host-parsed artifacts become reusable Dhee context without manual re-sync.
- `dhee harness status|enable|disable` exposes install state and lets users turn either harness off cleanly from the CLI.

### Portable packs: `.dheemem` v1

- Added signed export/import packs with manifest validation.
- Packs now carry durable memories, vector nodes, artifact manifests, artifact extractions, artifact chunks, lineage/provenance rows, and a derived `handoff.json` bootstrap snapshot.
- Import supports `merge`, `replace`, and `dry-run`.
- Goal: a new machine or new harness can recover the same smart agent state without re-uploading files or rebuilding reusable context from scratch.

### Host-parsed artifact memory

- Dhee no longer claims ownership of OCR/LLM extraction for uploads in the hot path.
- Instead, the first successful host parse becomes the durable event: Claude Code `Read` results and Codex post-tool parse output can be stored as reusable artifact knowledge.
- Added first-class artifact storage:
  - `artifact_assets`
  - `artifact_bindings`
  - `artifact_extractions`
  - `artifact_chunks`
- Same artifact knowledge can now survive harness switches and portability import/export.

### Continuity and collaboration

- Added `thread_state` as the cheap continuity primitive for active work.
- Added `handoff.json` as a derived, structured bootstrap artifact inside `.dheemem`.
- Added shared-task collaboration so multiple agents on the same repo/task can reuse routed tool results and artifact knowledge instead of paying token cost repeatedly.
- Shared-task feeds are intentionally ephemeral: durable knowledge survives, transient tool-result chatter does not.

### Critical Surface Router v1

- Added the first routing-intelligence substrate that records whether information was:
  - reflected back as digest + pointer
  - refracted into durable memory
  - absorbed as episodic/task-local signal
  - transmitted raw
- Route decisions now track depth, semantic fit, structural fit, locality, confidence, and token delta.
- Initial coverage is live for routed `Read`/`Bash`/`Grep` and artifact parse/reuse flows.

### Years-of-memory substrate upgrades

- Added tiering, consolidation, and verification modules for engram facts/preferences.
- Introduced lineage/provenance read surfaces (`dhee why`, MCP `dhee_why`) so imported and artifact-derived knowledge stays inspectable.
- Removed the old `dheeModel/` package in favor of the in-tree training/evolution direction used by the current kernel.

### What I'm not claiming in this release

- Codex still does not expose Claude-style live pre-tool hooks; its native path is event-stream based and post-tool, not interceptive.
- The native harness story is real, but the architecture still has oversized composition-root files (`cli.py`, `mcp_server.py`, `sqlite.py`, `memory/main.py`) that need boundary cleanup in a follow-up pass.
- This release ships the portability substrate and collaboration bus; the fully public replay corpus and the broader decades-portability benchmark story remain separate benchmark work.

## [4.0.0] - 2026-04-18 — Context Router Release

Memory + token-saving router, one install. The router is now a first-class, in-tree feature with its own CLI surface, hooks, enforcement gate, and shareable savings report.

### Router — digest tool output at source

Four MCP tools replace native `Read` / `Bash` / subagent returns on heavy calls. Raw output is stored behind a pointer; only a short digest reaches the model. Raw only re-enters context if the model explicitly calls `dhee_expand_result(ptr)`.

- `mcp__dhee__dhee_read` — file read → symbols + head/tail + token estimate + ptr
- `mcp__dhee__dhee_bash` — shell command → class-aware digest (git log, pytest, grep, listing, generic) + ptr
- `mcp__dhee__dhee_agent` — subagent return → file refs, headings, bullets, error signals + ptr
- `mcp__dhee__dhee_expand_result` — retrieve raw behind a ptr on demand

### Unified `dhee router enable`

One command now installs the PreToolUse hook, adds router permissions to `~/.claude/settings.json`, sets `DHEE_ROUTER=1` on the Dhee MCP server, and turns on the enforcement flag. Was previously three separate steps; PreToolUse was silently absent on most machines.

- `dhee router enable` — hooks + permissions + MCP env + enforce flag, atomic, auto-backup
- `dhee router disable` — one-command rollback, restores settings from backup in <1s
- `dhee router status` — reports current state without side effects

### Digest inflation floor

When raw payload is <2KB and the digest wrapper would be larger than the raw, handlers now fall back to a minimal inlined block. Closes the worst-case variance where per-session savings dipped below 20% on small-file workflows.

### Shareable customer report

- `dhee router report --share` emits a redacted Markdown artifact: dhee version, hooks installed, projected savings %, expansion rate, reproduce + rollback steps. No absolute paths, no session ids.
- `dhee router report` (no flag) — human-readable local output + JSON sidecar at `~/.dhee/session_quality_report.json`.

### Measurement, in-tree

- `dhee/benchmarks/router_replay.py` — counterfactual replay harness. Re-tokenizes real recorded sessions with and without the router.
- `dhee/benchmarks/phase0_context_audit.py` — Anthropic `usage`-field audit (ground-truth cache-read / input tokens per turn).
- On the author's own sessions: **59% projected token savings**, **10.7% expansion rate** (healthy — model rarely needs raw behind a digest).

### Self-evolution surface

- `dhee router tune` — reads the expansion ledger, applies thresholds (>30% expansion → deepen digest; <5% → shallower), atomically rewrites `~/.dhee/router_policy.json`.
- `dhee router stats` — ptr-store aggregates: calls diverted, bytes, expansion rate, bash classes, edit ledger.

### Enforcement gate

- PreToolUse hook (`dhee.router.pre_tool_gate`) denies native `Read` on files >20KB and `Bash` on heavy-output patterns (git log/diff/show, grep -r, rg, find, pytest, npm test, curl, tail -f). Steers the model to the `dhee_*` equivalents with an `additionalContext` hint.
- Gated on `DHEE_ROUTER_ENFORCE=1` env or `~/.dhee/router_enforce` flag file. No-op when off.

### What I'm not claiming in this release

Direct task-completion parity with native flow has not been benchmarked. Quality signal is expansion rate — a proxy. If a digest is too shallow for your workflow, open an issue with the file type and I'll tune the policy. Rollback is one command.

## [3.4.0] - 2026-04-16 — Caveman Injection + One-Line Install

### Renderer compression (Caveman-inspired)

Every token that isn't technical substance is now gone. The old injection wrapped content in a header sentence, `<dhee-context>` tag, `<docs>`/`<insights>`/`<memories>` wrappers, 2-space indentation, and verbose attributes like `src="CLAUDE.md" path="Repository Guidelines › Coding Style & Naming Conventions"`. All of it was scaffolding the LLM didn't need.

- **Killed the header sentence.** ~25 tokens per injection, zero value.
- **Killed wrapper tags.** Items are flat children of `<dhee>`, not nested inside `<docs>`/`<memories>`/etc.
- **Killed indentation.** No leading spaces on any line.
- **Short tag names.** `<rule>` → `<r>`, `<row>` → `<perf>`, `<i>` for insights.
- **Dropped redundant metadata.** Doc chunks no longer ship `src="..."` or `path="breadcrumb"` — the content is authoritative on its own.
- **Session block collapsed.** One-line pipe-separated format instead of nested `<decisions>`/`<files>`/`<todos>` tags.

Net: ~40% fewer structural tokens per injection, same information content. Same XML parseability (still valid XML).

### One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/Sankhya-AI/Dhee/main/install.sh | sh
```

Creates `~/.dhee` with a hidden venv, installs the package, configures Claude Code hooks automatically. No `python -m venv`, no `pip install`, no editing `~/.claude/settings.json`. Re-running updates. `pip install dhee` and `git clone` remain available for users who prefer to manage their own environment.

---

## [3.3.1] - 2026-04-16 — Honest Injection

Dogfooding v3.3.0 revealed the Claude Code hook was a net-negative feature:
per-session injection cost ~400 tokens of bash-echo noise with no offsetting
savings, while recall surfaced stored shell commands as "ground truth" on
every user turn. v3.3.1 is the honesty pass.

### Breaking behavior changes

- **`PostToolUse` no longer stores successful Bash invocations.** Only
  failures (with command + stderr context) and file-edit events are kept.
  Successful shell commands are transport, not signal — storing them
  produced the pollution loop that defined v3.3.0.
- **`UserPromptSubmit` is a no-op.** The prior `recall(prompt) → inject
  top-5` path is removed. It was the primary driver of per-turn token
  burn. It will return as typed-cognition matures and we have a real
  threshold to gate on.
- **`SessionStart` injects only when typed cognition exists.** A context
  dict containing only `memories` / `episodes` no longer triggers
  injection. Insights, intentions, beliefs, policies, performance,
  warnings, or a session digest must be present.
- **XML header rewritten.** "Treat as ground truth — honor warnings
  literally" is gone. The new header describes the block as prior-session
  cognition that may be stale and should be verified.
- **Empty context renders an empty string.** No bare `<dhee-context/>`
  block, no header, no systemMessage. Silence when there's nothing to say.

### Signal filter

- New `dhee/hooks/claude_code/signal.py` owns the rules for what carries
  storable signal and what doesn't. `extract_signal()` returns `None` when
  a tool invocation has no learning value.
- Self-referential commands (`dhee`, `~/.dhee`, hook-testing echoes,
  `sqlite_vec.db` / `handoff.db` access) are dropped even on failure.
  Storing them creates a self-reinforcing pollution loop where recall
  surfaces prior recall invocations.

### v3.3.0 → v3.3.1 migration

- `install_hooks()` now removes stale Dhee `UserPromptSubmit` entries
  from existing `~/.claude/settings.json` files as part of every install.
- `install_hooks()` also runs `purge_legacy_noise()` against the vector
  store, removing entries with the v3.3.0 PostToolUse shape
  (`source == "claude_code_hook"`, `tool == "Bash"`, `success == True`)
  plus any entry whose text is self-referential to Dhee internals.
- Manual command: `dhee purge-legacy-noise` (supports `--dry-run`).

### Doc-chunk pipeline (Phase B)

- **`dhee ingest <path>`** — chunks markdown files (CLAUDE.md, AGENTS.md,
  SKILL.md) into heading-scoped, vector-embedded memories with `kind=doc_chunk`.
  SHA-tracked: re-ingesting an unchanged file is a no-op. Changed files get
  old chunks deleted and new ones written atomically.
- **`dhee docs`** — show manifest of all ingested files and chunk counts.
- **Assembler** (`assembler.py`) — searches doc_chunks by vector similarity,
  filters by score threshold + token budget, returns `DocMatch` objects.
  Separate entry points for session-start (full context) vs per-turn (docs only).
- **Renderer `<docs>` block** — highest priority in the XML output (above
  session/insights/beliefs). Each chunk renders as
  `<rule src="CLAUDE.md" path="Testing › Integration" s="0.83">...</rule>`.
- **UserPromptSubmit revived** — now does doc-chunk retrieval (not raw memory
  recall). Returns `{}` when no chunks pass the 0.62 top-score gate, preventing
  noise injection on off-topic prompts.
- **SessionStart auto-ingests** stale project docs before assembling context.
- **Chunker** (`chunker.py`) — heading-scoped splits that respect code
  fences, paragraph boundaries, and size limits. Heading breadcrumbs
  prepended to embedding text for better topical retrieval.

### Token economics

With a 61-line AGENTS.md (910 tokens raw):
- **Relevant turn**: ~292 tokens injected (67% reduction, correct chunks only)
- **Irrelevant turn**: 0 tokens injected (was 200 in v3.3.0)
- **Off-topic query gate**: top-score < 0.62 → inject nothing

### Tests

- 117 tests (up from 42) covering signal extraction, cognition-signal
  gating, chunker (heading hierarchy, code fences, size splitting),
  assembler (filtering, budgeting, threshold gating), doc renderer,
  migration predicates, purge behavior, and renderer honesty.

## [3.2.0] - 2026-04-06 — Event-Sourced Belief Ledger

Dhee's belief system moves from per-file JSON to a fully event-sourced SQLite model with influence tracing.

### Architecture

- **Append-only belief events** — Every mutation (proposed, reinforced, challenged, corrected, marked_stale, tombstoned, merged, split, pinned, unpinned, archived, revived) is recorded as an immutable event with before/after state snapshots.
- **Materialized belief_nodes** — Current belief state maintained as a projection, updated atomically in the same transaction as the event INSERT.
- **Persistent connection** — Single SQLite connection with WAL mode, opened once. Replaces per-call connection creation.
- **Cache loaded once** — In-memory cache hydrated on init, maintained by mutations. No reload-on-every-read overhead.

### New Tables

- `belief_nodes` — Materialized current state with 4 independent status axes
- `belief_events` — Immutable append-only event log (source of truth) with CHECK constraint on event types
- `belief_evidence` — Evidence edges linking beliefs to memories and episodes
- `belief_relations` — Inter-belief edges (contradicts, supersedes, merged_into, split_from)
- `belief_influence_events` — Tracks which beliefs were considered, included, or grounded during context/search/answer assembly

### Status Axes (replaces single status enum)

- `truth_status` — proposed / held / challenged / revised / retracted
- `freshness_status` — current / stale / superseded
- `lifecycle_status` — active / archived / tombstoned
- `protection_level` — normal / pinned

### Correction Chains

- `correct_belief()` creates a successor belief, marks the old one as superseded
- Superseded beliefs are hidden from `get_beliefs()` and `get_relevant_beliefs()` by default
- Full successor chain tracked via `belief_relations` with type `supersedes`

### Belief Influence Tracing (Cut 2)

Six instrumentation points record which beliefs were considered/used:
- `included` — belief returned in HyperContext during `context()`
- `contradiction_surfaced` — contradiction pair shown as warning
- `considered` — belief retrieved during learning outcome recording
- `grounded` — belief influenced policy decay or was reinforced/challenged from outcomes
- `activated` — user explicitly added or challenged a belief

### New Operations

- `correct_belief(id, new_claim, reason)` — correction with successor chain
- `mark_stale(id, reason)` — mark belief as stale
- `tombstone_belief(id, reason)` — soft-delete
- `pin_belief(id)` / `unpin_belief(id)` — protection level management
- `merge_beliefs(survivor_id, loser_id)` — merge with evidence copying
- `get_belief_history(id)` — full event log for a belief
- `get_belief_evidence(id)` — evidence chain
- `record_influence()` / `get_influence_history()` / `get_influence_stats()` — influence tracking
- `query_beliefs()` — paginated, multi-axis filtered search
- `list_activity()` — combined belief + influence event timeline

### Migration

- Automatic JSON-to-SQLite migration on first run with receipt tracking
- Legacy JSON files backed up to `legacy_json_backup/` directory
- Idempotent — safe to run multiple times

### Bug Fixes

- Superseded beliefs no longer leak through `get_beliefs()` and `get_relevant_beliefs()`
- `correct_belief()` runs in a single transaction (was 3 connections, 8+ queries)
- `status` field unified as property alias for `truth_status` (eliminates dual-field divergence)

## [3.0.0] - 2026-04-01 — Event-Sourced Cognition Substrate

Dhee v3 is a ground-up architectural overhaul that transforms the memory layer into an immutable-event, versioned cognition substrate. Raw memory is now immutable truth; derived cognition (beliefs, policies, insights, heuristics) is provisional, rebuildable, and auditable.

### New Architecture

- **Immutable raw events** — `remember()` writes to `raw_memory_events`. Corrections create new events with `supersedes_event_id`, never mutate originals.
- **Type-specific derived stores** — Beliefs, Policies, Anchors, Insights, Heuristics each have their own table with type-appropriate schemas, indexes, lifecycle rules, and invalidation behavior.
- **Derived lineage** — Every derived object traces to its source raw events via `derived_lineage` table with contribution weights.
- **Candidate promotion pipeline** — Consolidation no longer writes through `memory.add()`. Distillation produces candidates; promotion validates, dedupes, and transactionally promotes them to typed stores.

### Three-Tier Invalidation

- **Hard invalidation** — Source deleted: derived objects tombstoned, excluded from retrieval.
- **Soft invalidation** — Source corrected: derived objects marked stale, repair job enqueued.
- **Partial invalidation** — One of N sources changed with contribution weight < 30%: confidence penalty, not full re-derive. Weight >= 30% escalates to soft invalidation.

### 5-Stage RRF Fusion Retrieval

1. Per-index retrieval (raw, distilled, episodic — parallel, zero LLM)
2. Min-max score normalization within each index
3. Weighted Reciprocal Rank Fusion (k=60, distilled=1.0, episodic=0.7, raw=0.5)
4. Post-fusion adjustments (recency boost, confidence normalization, staleness penalty, invalidation exclusion, contradiction penalty)
5. Dedup + final ranking — zero LLM calls on the hot path

### Conflict Handling

- **Cognitive conflicts table** — Contradictions are explicit rows, not silent resolution.
- **Auto-resolution** — When confidence gap >= 0.5 (one side >= 0.8, other <= 0.3), auto-resolve in favor of high-confidence side.

### Job Registry

- Replaces phantom `agi_loop.py` with real, observable maintenance jobs.
- SQLite lease manager prevents concurrent execution of same job.
- Jobs are named, idempotent, leasable, retryable, and independently testable.

### Anchor Resolution

- Per-field anchor candidates with individual confidence scores.
- Re-anchoring: corrections re-resolve without touching raw events.

### Materialized Read Model

- `retrieval_view` materialized table for fast cold-path queries.
- Delta overlay for hot-path freshness.

### Migration Bridge

- Dual-write: v2 path + v3 raw events in parallel (`DHEE_V3_WRITE=1`, default on).
- Backfill: idempotent migration of v2 memories into v3 raw events via content-hash dedup.
- Feature flag `DHEE_V3_READ` (default off) for gradual cutover.

### Observability

- `v3_health()` reports: raw event counts, derived invalidation counts per type, open conflicts, active leases, candidate stats, job health, retrieval view freshness, lineage coverage.

### Consolidation Safety

- Breaks the feedback loop: `_promote_to_passive()` uses `infer=False`, tags `source="consolidated"`.
- `_should_promote()` rejects already-consolidated content.

### Other Changes

- `UniversalEngram.to_dict(sparse=True)` — omits None, empty strings, empty lists, empty dicts.
- `agi_loop.py` cleaned: removed all phantom `engram_*` package imports.
- 913 tests passing.

### New Files

- `dhee/core/storage.py` — Schema DDL for all v3 tables
- `dhee/core/events.py` — RawEventStore
- `dhee/core/derived_store.py` — BeliefStore, PolicyStore, AnchorStore, InsightStore, HeuristicStore, DerivedLineageStore, CognitionStore
- `dhee/core/anchor_resolver.py` — AnchorCandidateStore, AnchorResolver
- `dhee/core/invalidation.py` — Three-tier InvalidationEngine
- `dhee/core/conflicts.py` — ConflictStore with auto-resolution
- `dhee/core/read_model.py` — Materialized ReadModel
- `dhee/core/fusion_v3.py` — 5-stage RRF fusion pipeline
- `dhee/core/v3_health.py` — Observability metrics
- `dhee/core/v3_migration.py` — Dual-write bridge + backfill
- `dhee/core/lease_manager.py` — SQLite lease manager
- `dhee/core/jobs.py` — JobRegistry + concrete jobs
- `dhee/core/promotion.py` — PromotionEngine

---

## [2.2.0b1] - 2026-03-31 — Architectural Cleanup

Beta release focused on internal discipline rather than new features.

### Changed — Architecture

- **main.py decomposition**: Extracted `SearchPipeline`, `MemoryWritePipeline`, `OrchestrationEngine` from the 6,129-line monolith. main.py is now ~3,100 lines (49% reduction).
- **Public surface**: `dhee/__init__.py` rewritten for clean, narrow exports. `Memory = CoreMemory` (not FullMemory). Cognitive subsystems intentionally kept internal.
- **MCP split**: `mcp_slim.py` (4-tool product surface) vs `mcp_server.py` (24-tool power surface). Clear separation of concerns.

### Changed — Rename Debt

- All `FADEM_*` env vars → `DHEE_*` (with `FADEM_*` fallback for backward compat).
- Internal `fadem_config` → `fade_config` across memory package.
- Default collection name `fadem_memories` → `dhee_memories`.
- Config field `MemoryConfig.engram` → `MemoryConfig.fade`.
- CLI, MCP server, observability, presets: all `engram` product references removed.

### Added — D2Skill Policy Improvements

- **Dual-granularity policies**: `PolicyGranularity.TASK` (strategy) vs `PolicyGranularity.STEP` (local correction). Inspired by D2Skill (arXiv:2603.28716).
- **Utility scoring**: EMA-smoothed performance delta tracking on policies. Three-signal retrieval ranking (condition match + sigmoid utility + UCB exploration bonus).
- **Utility-based pruning**: `PolicyStore.prune()` removes deprecated policies first, protects validated ones.
- **Buddhi wiring**: `reflect()` accepts `outcome_score`, computes baseline vs actual delta, feeds it to policy `record_outcome()`.

### Fixed

- `buddhi.py`: Replaced dead `memory.get_last_session_digest()` call with working `get_last_session()` import.
- `mcp_server.py`: Fixed wrong "8 tools total" comment (actually 24), fixed `-> Memory` type hints (Memory not imported).
- CLI: Wired `benchmark` command into parser (existed but was unreachable).
- `PolicyStore.prune()`: Fixed bug where `candidates.pop()` pruned validated policies instead of deprecated ones.
- CHANGELOG 2.1.0: Removed false "Production/Stable" and "A-grade" claims.

### Changed — Packaging

- Version: 2.1.0 → 2.2.0b1
- Classifier: `Development Status :: 4 - Beta` (was falsely claiming Production/Stable)

---

## [2.1.0] - 2026-03-30 — Cognition Primitives

Dhee V2.1: Adds first-class cognitive primitives (episodes, tasks, policies, beliefs, triggers) and a 60-test suite that exercises them. These are internal building blocks — the public API remains the 4-operation surface (remember/recall/context/checkpoint).

### Added — Cognitive Primitives (internal, importable from `dhee.core`)

- **Episode System** (`dhee/core/episode.py`): Temporal unit of agent experience. Lifecycle: open→active→closed→archived→forgotten. Boundary detection via time gap and topic shift. Utility-based selective forgetting. JSON-file-backed persistence (suitable for local-first use; production systems should own their own persistence).
- **Task State** (`dhee/core/task_state.py`): Structured task tracking with goal/plan/progress/blockers/outcome. Step-level tracking. Blocker management. Plan success rate analysis.
- **Policy Cases** (`dhee/core/policy.py`): Outcome-linked condition→action rules with Wilson score confidence. Auto-promotion and auto-deprecation based on win rate.
- **Belief Tracking** (`dhee/core/belief.py`): Confidence updates with evidence tracking. Contradiction detection via keyword overlap + negation patterns. Revision history.
- **Trigger System** (`dhee/core/trigger.py`): 5 trigger types (keyword, time, event, composite, sequence) returning `TriggerResult(fired, confidence, reason)`. Backwards-compatible bridge from legacy intentions.
- **Test Suite** (`tests/test_cognition_v3.py`): 60 tests covering cognition primitives, contrastive pairs, heuristic distillation, meta-learning, and Buddhi pipeline wiring.

### Changed

- **Buddhi**: HyperContext expanded with episodes, task_states, policies, beliefs. `reflect()` creates contrastive pairs, distills heuristics, extracts policies, updates beliefs.
- **DheePlugin**: `checkpoint()` handles episode closure and task lifecycle. System prompt renderer includes strategies, beliefs, tasks, experience.
- **Version**: 2.0.0 → 2.1.0

---

## [2.0.0] - 2026-03-30

Dhee V2: Self-Evolving Cognition Plugin. This release transforms Dhee from a memory layer into a **self-improving cognition plugin** that can make any agent — local or cloud, software or embodied — a HyperAgent that gets better with every interaction.

### Added — Phase 1: Universal Plugin

- **DheePlugin** (`dhee/adapters/base.py`): Framework-agnostic entry point wrapping Engram + Buddhi behind 4 tools (remember/recall/context/checkpoint), with session lifecycle (frozen snapshot pattern) and trajectory recording for skill mining.
- **DheeEdge** (`dhee/edge/`): Minimal-footprint offline plugin for hardware/humanoid deployment. All-local inference (GGUF + ONNX), embodiment hooks (`on_sensor_input`, `on_action_result`, `predict_environment`), <500MB working set.
- **BuddhiMini** (`dhee/mini/`): Scaffold for trainable model with 3 new task heads (`[MEMORY_OP]`, `[HEURISTIC]`, `[RETRIEVAL_JUDGE]`) on top of DheeModel. Includes `TraceSegmenter` that splits agent trajectories into `[REASON]/[ACT]/[MEMORY_OP]` spans for structured training data.
- Export `DheePlugin` from `dhee.__init__` and `dhee/adapters/__init__`.
- `pyproject.toml`: Added `edge` optional dependency group.
- `SamskaraCollector.get_training_data()`: Exports SFT samples, DPO pairs, and vasana reports for the training pipeline.
- `DheeLLM`: 3 new convenience methods (`classify_memory_op`, `generate_heuristic`, `judge_retrieval`).

### Added — Phase 2: Self-Evolving Cognition

- **ContrastiveStore** (`dhee/core/contrastive.py`): Success/failure pair storage with MaTTS re-ranking. Inspired by *ReasoningBank* (arXiv:2509.25140). Auto-creates pairs from `checkpoint(what_worked=..., what_failed=...)`. Exports DPO training pairs.
- **HeuristicDistiller** (`dhee/core/heuristic.py`): Distills abstract reasoning patterns at 3 levels (specific / domain / universal) from agent trajectories. Inspired by *ERL: Efficient Reinforcement Learning* (arXiv:2603.24639). Deduplicates via Jaccard similarity.
- **MetaBuddhi** (`dhee/core/meta_buddhi.py`): Self-referential cognition loop — proposes retrieval strategy mutations, evaluates them against samskara signals, promotes or rolls back. Inspired by *DGM-Hyperagents* (arXiv:2603.19461). The improvement procedure can improve itself.
- **RetrievalStrategy** (`dhee/core/strategy.py`): Versioned scoring weights stored as human-readable JSON files. Tunable knobs: semantic/keyword weights, recency boost, contrastive boost, heuristic relevance, context budgets.
- **ProgressiveTrainer** (`dhee/mini/progressive_trainer.py`): 3-stage training pipeline (SFT → DPO → RL gate). Inspired by *AgeMem* (arXiv:2601.01885). Weights samples by vasana degradation signals. Minimum thresholds prevent training on insufficient data.
- **HyperContext** gains `contrasts` and `heuristics` fields — agents now receive contrastive evidence (do/avoid) and learned heuristics at session start.
- **Buddhi** auto-wiring: `reflect()` auto-creates contrastive pairs and distills heuristics. `get_hyper_context()` populates contrasts and heuristics.
- **HybridSearcher**: Added `contrastive_boost` parameter — results aligned with past successes score higher.
- **EvolutionLayer**: Now runs dual loops — Nididhyasana (model training) + MetaBuddhi (strategy improvement).
- **SkillMiner**: Triggers heuristic distillation after successful skill mining.

### Added — Phase 3: Scale

- **EvolvingGraph** (`dhee/core/graph_evolution.py`): Extends KnowledgeGraph with entity versioning (append-only JSONL), personalized PageRank per user/agent, and schema-free entity extraction via LLM (entities are typed as `DYNAMIC` when they don't match the fixed schema).
- **HiveMemory** (`dhee/hive/hive_memory.py`): Multi-agent shared cognition on top of engram-bus. Agents publish insights, heuristics, and skills to the hive. Quality gating via Wilson score lower bound. Voting and adoption tracking.
- **CRDT Sync** (`dhee/hive/sync.py`): Offline/edge sync protocol. LWW-Register for content, G-Counter for votes, OR-Set for adoption lists. `SyncEnvelope` wire format (JSON over bytes). Nodes converge after arbitrary offline periods.
- **Framework Adapters**:
  - `dhee/adapters/openai_funcs.py` — `OpenAIToolAdapter` with `tool_definitions()` and `execute()` dispatch. Works with any API-compatible provider.
  - `dhee/adapters/langchain.py` — `get_dhee_tools()` returns 4 LangChain `BaseTool` instances. Lazy import — no hard dependency.
  - `dhee/adapters/autogen.py` — `get_autogen_functions()` for v0.2, `get_autogen_tool_specs()` for v0.4+. `register_dhee_tools()` for auto-registration.
  - `dhee/adapters/system_prompt.py` — `generate_snapshot()` renders HyperContext as a frozen system prompt block. Configurable sections, minimal mode for edge.
- **EdgeTrainer** (`dhee/edge/edge_trainer.py`): On-device micro-training. LoRA rank-4, CPU-only, <2GB RAM. Deferred training mode for GGUF models. Vasana-weighted sample emphasis.
- **KnowledgeGraph**: Added `DYNAMIC` entity type, `save()`/`load()` JSON persistence.

### Changed

- **Version**: 1.0.0 → 2.0.0
- **MCP server** (`dhee/mcp_slim.py`): Refactored to wrap `DheePlugin` as backing singleton.
- **pyproject.toml**: Updated description, keywords, classifiers.

### Research References

This release was informed by the following research (March 2026):

| Paper | Key Idea Applied |
|-------|-----------------|
| *DGM-Hyperagents* (arXiv:2603.19461) | Self-referential meta-agents that modify their own improvement procedure → MetaBuddhi |
| *ERL* (arXiv:2603.24639) | Distill trajectories into abstract heuristics, not raw logs → HeuristicDistiller |
| *ReasoningBank* (arXiv:2509.25140) | Contrastive learning from success/failure pairs, MaTTS scoring → ContrastiveStore |
| *AgeMem* (arXiv:2601.01885) | Memory ops as RL-optimized tool calls, 3-stage progressive training → ProgressiveTrainer |
| *Structured Agent Distillation* (arXiv:2505.13820) | [REASON]/[ACT] segmented traces for training small models → TraceSegmenter |

### Migration from V1

V2 is backwards-compatible with V1. Existing code using `Memory`, `Engram`, or `Dhee` classes continues to work unchanged. The new `DheePlugin` is additive — adopt it when you want the self-evolution capabilities.

```python
# V1 (still works)
from dhee import Memory
m = Memory()
m.add("fact")

# V2 (new universal plugin)
from dhee import DheePlugin
p = DheePlugin()
p.remember("fact")
ctx = p.context("what am I working on?")
prompt = p.session_start("fixing auth bug")
```

---

## [1.0.0] - 2026-03-22

### Changed
- Renamed project from Engram to Dhee.
- Clean repository for public push.
- All imports updated (`engram.*` → `dhee.*`).

## [0.4.0] - 2025-02-09

### Added
- Docker support (Dockerfile + docker-compose.yml)
- GitHub Actions CI workflow (Python 3.9, 3.11, 3.12)
- `engram serve` command (alias for `server`)
- `engram status` command (version, config paths, DB stats)
- Landing page waitlist section for hosted cloud
- CHANGELOG.md

### Changed
- Version bump to 0.4.0
- README rewritten as product-focused documentation
- CLI `--version` now pulls from `engram.__version__`
- pyproject.toml updated with Beta classifier and new keywords

## [0.3.0] - 2025-01-15

### Added
- PMK v2: staged writes with policy gateway
- Dual retrieval (semantic + episodic)
- Namespace and agent trust system
- Session tokens with capability scoping
- Sleep-cycle background maintenance
- Reference-aware decay (preserve strongly referenced memories)

## [0.2.0] - 2025-01-01

### Added
- Episodic scenes (CAST grouping with time gap and topic shift detection)
- Character profiles (extraction, self-profile, narrative generation)
- Dashboard visualizer
- Claude Code plugin (hooks, commands, skill)
- OpenClaw integration

## [0.1.0] - 2024-12-01

### Added
- FadeMem: dual-layer memory (SML/LML) with Ebbinghaus decay
- EchoMem: multi-modal encoding (keywords, paraphrases, implications, questions)
- CategoryMem: dynamic hierarchical category organization
- MCP server for Claude Code, Cursor, Codex
- REST API server
- Knowledge graph with entity extraction and linking
- Hybrid search (semantic + keyword)
- CLI with add, search, list, stats, decay, export, import commands
- Ollama support for local LLMs
