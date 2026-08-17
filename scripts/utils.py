"""The machinery every stage shares: identifiers, files, validation, reporting,
and the loop that both generation and judging run.

Nothing here states any part of the design. Everything it needs it takes from
settings.py, so a revision to the design reaches this module without it being
edited.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pandas as pd
from settings import (BENCHMARK_COLUMNS, DATASET_NAMES, DATA_DIRS, DOMAIN_NAMES,
                      DRAFTS_COLUMNS, ENV_PATH, FORMS, PROVIDER_KEYS, TYPES)

# ----------------------------------------------------------------------------
# Identifiers
# ----------------------------------------------------------------------------

# Define function to build a scenario identifier
def make_scenario_id(code, scenario_type, index):
    return f'{code}-{TYPES[scenario_type]["code"]}{index}'


# Define function to build a source identifier
def make_source_id(dataset, record_id):
    return f'{dataset}-{record_id}'


# Define function to build a prompt identifier
def make_prompt_id(scenario_id, condition):
    return f'{scenario_id}-{condition}'


# Define function to read the domain code back out of a scenario identifier
def code_from_scenario(scenario_id):
    return str(scenario_id).split('-')[0]


# Define function to place the age signal in front of the scenario request
def make_prompt(opener, request):
    return f'{opener} {request}'.strip()


# Define function to turn a model identifier into a filename. A slash separates
# an owner from a model on the Hugging Face hub and a colon separates a tag in
# Ollama, and neither belongs in a path.
def model_slug(model_id):
    return re.sub(r'[^a-z0-9.-]+', '-', str(model_id).lower()).strip('-')


# Define function to name the file one model writes to in one experiment
def result_path(model_id, directory):
    return directory / f'{model_slug(model_id)}.jsonl'


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------

# Define function to make the data directories, so a script can write without
# each one checking first
def make_directories(directories=DATA_DIRS):
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Define function to read a table of text, leaving every field as it was written
def read_table(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna('')


# Define function to read a JSON lines file, which is how model output is kept:
# a reply can hold newlines and quotes that CSV quoting mishandles, and a line
# appended as each reply arrives survives a run that stops part way
def read_lines(path):
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


# Held while a line is written. Once calls run several at a time, two workers can
# reach this at once, and a reply of several kilobytes is past the size at which
# an append arrives whole. Without this, two half records interleave into one
# line that no later stage can parse.
_APPENDING = Lock()


# Define function to append one record to a JSON lines file
def append_line(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + '\n'
    with _APPENDING:
        with path.open('a') as file:
            file.write(line)


# Define function to read every model's output in one directory as one frame
def read_all(directory):
    frames = [read_lines(path) for path in sorted(directory.glob('*.jsonl'))]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Define function to read the api key a provider expects, from .env
def api_key(provider):
    variable = PROVIDER_KEYS.get(provider)
    return environment(variable) if variable else ''


# Define function to read one value from .env or the environment
def environment(name):
    if name not in os.environ and ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            key, _, value = line.partition('=')
            if key.strip() and not key.strip().startswith('#'):
                os.environ.setdefault(key.strip(), value.strip())
    return os.environ.get(name, '')


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

_SECTIONS = []


# Define function to head a block of printed output
def section(title):
    print(f'\n{title}' if _SECTIONS else title)
    _SECTIONS.append(title)


# Define function to describe the shape of a table
def shape_of(frame):
    return f'{len(frame)} rows, {frame.shape[1]} columns'


# Define function to report validation problems and stop when any are found
def report(name, problems):
    if not problems:
        print(f'Validated {name}')
        return
    for problem in problems:
        print(f'  {name}: {problem}')
    raise SystemExit(f'{len(problems)} validation problems in {name}')


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

# Define function to keep only the scenario rows that have been written
def written(scenarios):
    return scenarios[scenarios['request'].str.strip() != ''].reset_index(drop=True)


# Define function to check a table for the faults that break the pipeline
def validate(frame, required, id_column='', text_columns=(), labels=None):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return [f'missing columns {", ".join(missing)}']

    problems = []
    if id_column:
        repeated = frame[id_column][frame[id_column].duplicated()].unique()
        if len(repeated):
            problems.append(f'{len(repeated)} duplicate ids, first {repeated[0]}')
    for column in text_columns:
        blank = frame[column].fillna('').astype(str).str.strip() == ''
        if blank.any():
            problems.append(f'{int(blank.sum())} empty values in {column}')
    for column, allowed in (labels or {}).items():
        values = frame[column].fillna('').astype(str)
        invalid = sorted(set(values) - {str(value) for value in allowed})
        if invalid:
            problems.append(f'invalid {column} labels {", ".join(invalid[:5])}')
    return problems


# Define function to check the drafts a scenario may be selected from
def check_drafts(drafts):
    problems = validate(frame=drafts, required=DRAFTS_COLUMNS,
                        id_column='source_id',
                        text_columns=['source_id', 'domain'],
                        labels={'domain': DOMAIN_NAMES.values(),
                                'dataset': DATASET_NAMES + [''],
                                'scenario_type': TYPES,
                                'form': FORMS + ['']})
    if problems:
        return problems
    return problems


# Define function to check the written scenarios
def check_benchmark(scenarios):
    return validate(frame=scenarios, required=BENCHMARK_COLUMNS,
                    id_column='scenario_id',
                    text_columns=['scenario_id', 'request'],
                    labels={'scenario_type': TYPES,
                            'domain': DOMAIN_NAMES.values(),
                            'dataset': DATASET_NAMES + [''],
                            'form': FORMS})


# ----------------------------------------------------------------------------
# The loop generation and judging share
# ----------------------------------------------------------------------------

# How often to report, in seconds. A line per call would run to tens of
# thousands of lines and bury the failures worth seeing.
REPORT_EVERY = 60

# How many api calls to have in flight at once. A live pass spends nearly all of
# its time waiting on a reply rather than sending one, so sending several at
# once costs nothing extra and finishes in a fraction of the time. Kept modest,
# because a provider that rate limits will refuse the surplus rather than queue
# it, and the retry that follows wastes more than the concurrency gained.
WORKERS = 12

# Define function to list the items of one file still to collect. An item is
# identified by its keys alone, and a row already written for those keys is not
# repeated unless it failed in transit.
def outstanding(wanted, collected, keys):
    if collected.empty:
        return wanted
    done = {tuple(str(row[key]) for key in keys)
            for _, row in collected.iterrows() if not str(row['error']).strip()}
    return [item for item in wanted
            if tuple(str(item[key]) for key in keys) not in done]


# Define function to report what is outstanding before a stage begins
def announce(path, wanted, pending, limit=0):
    print(f'{len(wanted) - len(pending)} of {len(wanted)} already collected '
          f'in {path.name}')
    if limit:
        pending = pending[:limit]
    if not pending:
        return []
    print(f'{len(pending)} to collect now')
    return pending


# Define function to work through the outstanding items, appending as they
# arrive, so that a run stopping part way loses nothing and resumes where it
# left off. A running cost is reported where the model is billed per token, so
# that a pass can be stopped before it overruns a budget rather than after.
#
# Three ways of getting through the list, and which one applies is a property of
# the backend. produce_batch hands a group to a runtime that schedules them
# together, which is what vLLM is for. workers above one puts them to an api
# concurrently, which is worth it when each call spends its time waiting rather
# than computing. Otherwise one at a time.
#
# Results are appended as they finish, so with workers the file order is the
# order they came back rather than the order asked. Nothing reads that order:
# every record carries its own prompt and replicate, and resume matches on those.
def collect(pending, produce, path, label='', meter=None, produce_batch=None,
            batch_size=1, workers=1):
    started, spoke, failures = time.time(), time.time(), 0
    size = batch_size if produce_batch else 1
    index = 0

    def attempt(item):
        try:
            return produce(item), ''
        except Exception as problem:
            # one failure should not end an overnight run, so it is recorded and
            # the pass continues; a rerun retries whatever failed
            return {}, f'{type(problem).__name__}: {problem}'

    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(0, len(pending), max(size, workers if pool else 1)):
            group = pending[start:start + max(size, workers if pool else 1)]
            if produce_batch:
                try:
                    results = produce_batch(group)
                    errors = [''] * len(group)
                except Exception as problem:
                    results = [{}] * len(group)
                    errors = [f'{type(problem).__name__}: {problem}'] * len(group)
                    failures += len(group)
            elif pool:
                # each call is independent, so they are sent together and written
                # in the order they were asked for rather than the order they land
                outcomes = list(pool.map(attempt, group))
                results = [result for result, _ in outcomes]
                errors = [error for _, error in outcomes]
                failures += sum(1 for error in errors if error)
            else:
                result, error = attempt(group[0])
                results, errors = [result], [error]
                failures += 1 if error else 0

            for item, result, error in zip(group, results, errors):
                append_line(path, {**item, **result, 'error': error})
            index += len(group)

            if time.time() - spoke >= REPORT_EVERY or index == len(pending):
                spoke = time.time()
                rate = index / max(time.time() - started, 1)
                line = (f'  {label + "  " if label else ""}{index} of {len(pending)}, '
                        f'{rate * 3600:.0f} an hour, '
                        f'{(len(pending) - index) / rate / 3600:.1f} hours left, '
                        f'{failures} failed')
                if meter is not None:
                    spent, tokens = meter()
                    if spent is not None:
                        projected = spent / index * len(pending)
                        line += (f'\n     ${spent:,.4f} spent, ${projected:,.2f} '
                                 f'projected for this pass, {tokens:,} tokens')
                print(line)
    finally:
        if pool:
            pool.shutdown()
    return failures
