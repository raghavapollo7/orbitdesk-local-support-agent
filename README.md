# OrbitDesk Local-First Support Agent

This project is a local-only support agent for the supplied fictional OrbitDesk product. It uses LangGraph for routing, a local Hugging Face embedding model for retrieval, and a local Hugging Face language model for response generation. It does not call hosted language-model APIs.

## Architecture

```text
START -> Triage
  answerable -> Retrieval -> Generation -> Verification -> Finalize
                                                 |
                                                 +-> Revision -> Verification (once)
  requires_clarification -> Clarification response -> END
  requires_escalation -> Escalation response -> END
  out_of_scope -> Safe refusal -> END
```

`AgentState` is a TypedDict containing the query, classification, retrieved documents, answer, verification result, retry count, and node log. Every node appends its name to `node_log`, making executed routes visible in CLI output.

The retry guard is explicit: verification routes to `revision` only while `retry_count < 1`. A second verification failure returns `safe_failure`, so the graph cannot loop indefinitely.

## Engineering choices

- Retrieval: `sentence-transformers/all-MiniLM-L6-v2` builds an in-memory NumPy cosine-similarity index over the Markdown knowledge base.
- Generation: `Qwen/Qwen2.5-0.5B-Instruct` is a small instruction model that is practical on CPU for this assignment.
- Triage: deterministic rule-hybrid logic handles known unsafe, escalation, and ambiguous patterns. This keeps high-risk routing inspectable.
- Verification: deterministic checks validate source presence, dangerous claims or secret requests, and lexical evidence overlap. If an initial draft fails, the revision node produces a short evidence-grounded correction for the documented answerable paths before re-verification. This makes the retry path reproducible instead of asking a model to judge itself.

The verifier checks non-empty answers, source-reference shape, prohibited secret or unsupported-action language, common unsupported boilerplate, response length, and lexical overlap with the retrieved evidence. The final CLI response uses the supplied output-schema fields only; execution logs and model metrics print separately. The CLI prints resolved model revisions, load time, and generation device. Record those values, plus hardware details, in the submission form after running the application.

### Recorded run environment

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Generation model: `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`
- Hardware: AMD Ryzen 7 7840HS, 16 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU, and AMD Radeon 780M integrated graphics
- Demonstrated device: CPU

## Setup

Python 3.12 is recommended for Windows PyTorch compatibility.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first run downloads the two Hugging Face models. Later, use `--offline` to demonstrate that the application works without network access.

## Run

The supplied assignment material is included under `data/`.

```powershell
python -m orbitdesk_agent.cli "Our daily dashboard exports stopped after an Admin changed the workspace timezone." --offline
```

To show the verification retry route deliberately:

```powershell
python -m orbitdesk_agent.cli "Our daily dashboard exports stopped after an Admin changed the workspace timezone. What should we check, and can the missed export be recovered?" --offline --force-first-verification-failure
```

## Tests

```powershell
python -m pytest
```

The tests stub generation and assert the actual graph path. They do not depend on wording produced by a language model.

## Submission checklist

- Run the five questions in `data/sample_questions.json`.
- Capture the CLI's `node_log` for at least three distinct routes.
- Capture the forced verification failure and retry route.
- Upload `docs/graph.png` to the Google Form. The editable Mermaid source is `docs/graph.mmd`.
- Add the exact resolved model revisions and your CPU, RAM, and GPU details to the submission.

## Known limitation

The small local model may be less fluent than a hosted model. The system prioritizes grounded evidence, deterministic verification, and traceable routing over prose quality.

## AI assistance disclosure

AI coding assistance was used to help plan and implement this assignment. The architecture, local-only model constraint, routing behavior, tests, and final verification were reviewed and run by the submitter.
