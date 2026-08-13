# Scripts

Run in this order. Each reads the design from `config/` and reports what it did.

| Script | Reads | Writes |
|---|---|---|
| `download_data.py` | `config/datasets.yml` | `data/sources/` |
| `prepare_data.py` | sources, `config/scenarios.yml` | `data/benchmark/drafts.csv`, `benchmark.csv` |
| `build_data.py` | `benchmark.csv` | `prompts.csv` |
| `score_data.py` | `benchmark.csv` | `scores.csv` |
| `run_generation.py` | `prompts.csv` | `results/adaptation/` |
| `build_turns.py` | `prompts.csv`, replies | `results/persistence/` |

Two more are run by hand rather than in sequence. `run_test.py` puts one prompt
through the whole pipeline, which is how to check a model loads and the policy
reads before committing to a pass. `check_models.py` verifies that every
identifier in `config/models.yml` still resolves.

Four are modules rather than scripts. `settings.py` loads the configuration and
holds the paths, identifiers and validation every script shares. `backends.py`
generates one reply through whichever runtime is available. `judge.py` builds the
policy the classifier applies and reads its verdict back; running it prints the
policy. `fill_scenarios.py` expands `config/scenarios.yml` into the drafts.

`archive/` holds the lexical simplification that was tried and abandoned. It is
kept because Chapter 3 reports the attempt as a negative result, and is not part
of the pipeline.
