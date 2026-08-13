# Does Age Matter?

Code and data for a benchmark that asks whether a chatbot changes what it will
answer when it is told how old the user is. Every request is byte-identical
across the age conditions, so the age signal is the only thing that varies.

## Layout

```
config/
  settings.yml     the design, the corpora, and the model panel
  scenarios.yml    the 120 scenarios, the one part written by hand

scripts/
  settings.py      what the design states, and where everything lives
  utils.py         identifiers, files, validation, and the loop the stages share
  backends.py      one reply, through whichever runtime is available
  download.py      1  the corpora        -> data/sources/
  build.py         2  the benchmark      -> data/benchmark/
  run.py           3  the replies        -> results/adaptation/
  evaluate.py      4  the judgements     -> results/judgements.csv

data/              the corpora downloaded, and the benchmark built from them
notebooks/         what the pipeline produced, read back
jobs/              cluster submission
archive/           an approach tried and abandoned, kept because Chapter 3 reports it
```

## Setup

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
cp .env.example .env        # then add the api keys the panel needs
```

## Running it

Four stages, in order. Each reads the design from `config/`, writes into `data/`
or `results/`, and reports what it did.

| Stage | Command | Writes |
|---|---|---|
| 1. Corpora | `python scripts/download.py` | `data/sources/` |
| 2. Benchmark | `python scripts/build.py` | `drafts.csv`, `benchmark.csv`, `prompts.csv`, `scores.csv` |
| 3. Replies | `python scripts/run.py generate --model <id> --backend <runtime>` | `results/adaptation/<model>.jsonl` |
| 4. Judgements | `python scripts/evaluate.py --backend <runtime>` | `results/judgements.csv` |

Stage 2 is the whole build: it reads the corpora and `config/scenarios.yml`,
opens a draft per usable source record, fills the 120 scenario slots from the
drafts marked to keep, expands each across the eleven age conditions, and scores
every request variant for readability. `drafts.csv` is the only file edited by
hand, and work already in it is preserved on every run, so the stage can be
rerun after any revision to the design.

Stages 3 and 4 append one line at a time and skip whatever is already collected,
so a run that stops part way resumes where it left off. Give either a `--limit`
first: the rate it reports is enough to estimate a full pass before committing a
night to it.

The persistence extension is built once replies exist:

```
python scripts/build.py turns       # results/persistence/dialogues.csv
```

## Before a run

```
python scripts/run.py check
python scripts/run.py check --model qwen3:8b --backend ollama \
    --prompt-id sub-h1-age09
python scripts/evaluate.py --policy
```

The first looks every identifier up against its provider, so a renamed or
retired model fails here rather than part way through generation, and reports
how many calls the panel implies. Naming a model puts one prompt through
generation, scoring and comparison as well; `--reply` scores a supplied text
instead, which exercises the scoring path without loading a large model. The
last prints the policy as the classifier sees it.

## On a cluster

Compute nodes have no outbound network, so a model is fetched from a login node
first and the job runs offline against the cache.

```
bash jobs/fetch.sh Qwen/Qwen3.6-27B
qsub -v MODEL=Qwen/Qwen3.6-27B jobs/run.sh
qsub -v STAGE=judge jobs/run.sh
```

## The modules

Three files are imported rather than run. `settings.py` reads the two
configuration files and exposes what they state along with everything that
follows from it; it does no work. `utils.py` holds the machinery that acts on
those values: identifiers, file reading and writing, validation, reporting, and
the resumable loop that generation and judging both run. `backends.py` generates
one reply through vLLM on a GPU, Ollama or MLX locally, or transformers
anywhere. `evaluate.py` is both: it holds the policy, the verdict reading and
the language measures, which `build.py` also uses to score the requests, and
running it scores the replies.

## Licences

The code is under the licence in `LICENSE`. The source corpora keep their own,
recorded with a checksum per file in `data/sources/downloads.md`. They are
downloaded rather than redistributed here.
