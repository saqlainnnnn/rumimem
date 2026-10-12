# RumiMem

**RumiMem** is an experimental, Titans-inspired fast-weight neural memory for pretrained text language models. The project investigates whether a small neural memory can retain and update conversational information across delayed queries and intervening sessions.

> Research status: scaffold only. No claim of reproducing Titans or demonstrating persistent conversational recall has been made.

## Research questions

1. How many key–value associations can a small fast-weight network retain before interference becomes unacceptable?
2. Does delayed recall survive unrelated intervening sessions?
3. Can corrections replace stale information without damaging unrelated memories?
4. Does selective writing improve retention and reduce unnecessary updates?

## Planned milestones

- [ ] M0 — Reproducible project scaffold and environment
- [ ] M1 — Sequential Titans-style fast-weight update with unit tests
- [ ] M2 — Synthetic multi-session delayed-recall task and baselines
- [ ] M3 — Ablations: momentum, forgetting, write gate, shuffled updates
- [ ] M4 — Checkpoint/resume and per-user state isolation
- [ ] M5 — Small pretrained text-LLM integration

## Setup

Python 3.10+ is recommended. Start in a clean virtual environment.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

For Colab, install the project dependencies in a fresh runtime and save checkpoints to Google Drive. Keep experiments small, deterministic where possible, and resumable.

## Project layout

```text
rumimem/
├── configs/       # Explicit experiment configurations
├── docs/          # Architecture notes and experiment protocol
├── experiments/   # Small runnable experiment entry points
├── notebooks/     # Colab-oriented notebooks (kept lightweight)
├── src/rumimem/
│   ├── memory/    # Fast-weight memory and update rules
│   └── data/      # Synthetic episode generation
└── tests/         # Unit tests for update semantics and persistence
```

## Research integrity

- Titans is the architectural reference, not a claim that this project reproduces the complete Titans model.
- The sequential update is the correctness reference before any chunk-parallel implementation.
- Reconstruction loss alone is not evidence of useful long-term memory.
- Delayed-query evaluation must not expose the original supporting text in the ordinary model context.
- Report seeds, configuration, compute, baselines, failures, and limitations.

## Primary reference

Behrouz et al., [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663).

## License

MIT. See [LICENSE](LICENSE).
