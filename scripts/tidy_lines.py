"""Rewrites a JSON lines file so every record carries its fields in one order.

The generation script appends a record as each reply arrives, so a file written
across a change to that order holds both. Nothing downstream depends on the
order, since a record is read by key, but a consistent file is far easier to read
directly.

    python scripts/tidy_lines.py                       every file in data/responses
    python scripts/tidy_lines.py data/dialogues         every file in that folder
    python scripts/tidy_lines.py data/responses/qwen3-8b.jsonl   just the one
"""

import json
import sys
from pathlib import Path

from settings import RESPONSES_DIR

# The order fields are written in: what identifies the reply, what produced it,
# whether it worked, and the text itself last, so the short fields line up
ORDER = ['prompt_id', 'model', 'replicate', 'backend', 'temperature', 'error',
         'response']


# Define function to put one record's fields in order, keeping any extras
def reorder(record, order=ORDER):
    known = {key: record[key] for key in order if key in record}
    extra = {key: value for key, value in record.items() if key not in order}
    return {**known, **extra}


# Define function to rewrite a file in place, through a temporary copy
def tidy(path, order=ORDER):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    records = [reorder(json.loads(line), order) for line in lines]
    temporary = path.with_suffix(path.suffix + '.tidy')
    temporary.write_text(''.join(json.dumps(record) + '\n'
                                 for record in records))
    temporary.replace(path)
    return records


if __name__ == '__main__':
    given = Path(sys.argv[1]) if len(sys.argv) > 1 else RESPONSES_DIR
    if not given.exists():
        raise SystemExit(f'{given} not found')

    # a directory is tidied file by file, since replies are collected one file
    # per model rather than into a single stream
    paths = sorted(given.glob('*.jsonl')) if given.is_dir() else [given]
    if not paths:
        raise SystemExit(f'no .jsonl files in {given}')

    for path in paths:
        records = tidy(path)
        orders = {tuple(record) for record in records}
        print(f'{path.name}: {len(records)} records, '
              f'{len(orders)} field order{"" if len(orders) == 1 else "s"}')
