"""Collects replies to every prompt, one line at a time.

Each reply is appended to the model's file as it arrives, and any prompt already
there is skipped, so a run that stops part way resumes where it left off. That
matters because a full pass is tens of thousands of generations and will not
finish in one sitting.

    ollama pull qwen3:8b
    python scripts/run_generation.py --model qwen3:8b --backend ollama

    python scripts/run_generation.py --backend mlx --limit 60 \
        --model mlx-community/Qwen2.5-7B-Instruct-4bit

Run it with a limit first: the rate it reports is enough to estimate the whole
pass before committing a night to it. Progress prints once a minute rather than
per reply, so the log stays short enough to read the morning after.
"""

import argparse
import time

import pandas as pd
from backends import BACKENDS, ask
from settings import (ADAPTATION_DIR, GENERATION, PROMPTS_PATH, adaptation_path,
                      append_line, make_directories, read_all, read_lines,
                      section, shape_of)

# How often to report, in seconds. A line per reply would run to tens of
# thousands of lines and bury the failures worth seeing.
REPORT_EVERY = 60

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to list the prompt and replicate pairs still to collect
def outstanding(prompts, collected, replicates):
    wanted = [(row.prompt_id, replicate) for row in prompts.itertuples()
              for replicate in range(1, replicates + 1)]
    if collected.empty:
        return wanted
    # a reply that failed is not collected, so a rerun retries it
    done = {(row.prompt_id, int(row.replicate)) for row in collected.itertuples()
            if not str(row.error).strip()}
    return [pair for pair in wanted if pair not in done]


# Define function to collect the outstanding replies, appending as they arrive
def collect(prompts, pending, backend, model, max_tokens, temperature, path):
    by_id = prompts.set_index('prompt_id')['prompt']
    started, spoke, failures = time.time(), time.time(), 0
    for index, (prompt_id, replicate) in enumerate(pending, start=1):
        try:
            reply, error = ask(backend, model, by_id[prompt_id], max_tokens,
                               temperature), ''
        except Exception as problem:
            # one failure should not end an overnight run, so it is recorded and
            # the pass continues; a rerun retries whatever failed
            reply, error = '', f'{type(problem).__name__}: {problem}'
            failures += 1
        append_line(path, {'prompt_id': prompt_id, 'model': model,
                           'replicate': replicate, 'backend': backend,
                           'temperature': temperature, 'error': error,
                           'response': reply})
        if time.time() - spoke >= REPORT_EVERY or index == len(pending):
            spoke = time.time()
            rate = index / max(time.time() - started, 1)
            print(f'  {index} of {len(pending)}, {rate * 3600:.0f} an hour, '
                  f'{(len(pending) - index) / rate / 3600:.1f} hours left, '
                  f'{failures} failed')
    return failures


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', default='ollama', choices=list(BACKENDS))
    parser.add_argument('--replicates', type=int, default=GENERATION['replicates'])
    parser.add_argument('--max-tokens', type=int, default=GENERATION['max_tokens'])
    parser.add_argument('--temperature', type=float,
                        default=GENERATION['temperature'])
    parser.add_argument('--limit', type=int, default=0,
                        help='stop after this many replies, to time a pass')
    arguments = parser.parse_args()

    section('Generation')
    make_directories()
    prompts = pd.read_csv(PROMPTS_PATH, dtype=str, keep_default_na=False)
    path = adaptation_path(arguments.model)
    pending = outstanding(prompts=prompts, collected=read_lines(path),
                          replicates=arguments.replicates)

    wanted = len(prompts) * arguments.replicates
    print(f'{arguments.model} on {arguments.backend}, {len(prompts)} prompts '
          f'times {arguments.replicates} replicates')
    print(f'{wanted - len(pending)} of {wanted} already collected in {path.name}')
    if arguments.limit:
        pending = pending[:arguments.limit]
    if not pending:
        raise SystemExit('nothing outstanding')
    print(f'{len(pending)} to collect now')

    failures = collect(prompts=prompts, pending=pending,
                       backend=arguments.backend, model=arguments.model,
                       max_tokens=arguments.max_tokens,
                       temperature=arguments.temperature, path=path)

    section('Collected')
    everything = read_all(ADAPTATION_DIR)
    print(f'{shape_of(everything)} across {ADAPTATION_DIR.name}')
    print(everything.groupby('model').size().to_string())
    if failures:
        print(f'{failures} failed this pass, run again to retry them')
