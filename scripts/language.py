"""Measures the language of every collected reply.

    python scripts/language.py
    python scripts/language.py --model claude-haiku-4-5-20251001
    python scripts/language.py --floor 100

Reads results/adaptation/ and writes results/language/, one file per model.

These are computed from the text and need no model, which is why they are here
rather than in the judgement. A classification costs a call to a classifier and
cannot be revised without another one; readability costs arithmetic, so a change
of mind about how it is measured is a second rather than an afternoon.

    FKGL              Flesch Kincaid grade level
    FRE               Flesch reading ease
    Mean AoA          mean age of acquisition, from the Kuperman norms
    Response Length   words

A floor applies to the first three. Readability formulas are unstable on short
texts and meaningless on very short ones: 'I cannot help with that' has a grade
level below zero, which is arithmetic rather than a finding. Replies under the
floor are measured for length and left blank for the rest, and the share left
blank is reported per model, since it differs enormously between them and is
itself a result.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate
from settings import (ADAPTATION_DIR, LANGUAGE_COLUMNS, LANGUAGE_DIR,
                      measure_column)
from utils import append_line, make_directories, read_lines, result_path, section

# Words below which the readability formulas are not reported. A hundred is the
# usual floor for FKGL to be stable and fifty is where it stops being nonsense;
# fifty is used, so that the measure covers as much of the data as it can while
# still meaning something.
FLOOR = 50


# Define function to measure one reply, leaving readability blank when the reply
# is too short for the formulas to say anything
def measure(text, norms, floor=FLOOR):
    scored = evaluate.score_reply(text, norms)
    words = scored[measure_column('Response Length')]
    if words >= floor:
        return scored
    return {name: (words if name == measure_column('Response Length') else '')
            for name in scored}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='', help='one model, or all of them')
    parser.add_argument('--floor', type=int, default=FLOOR,
                        help='words below which readability is left blank')
    arguments = parser.parse_args()

    make_directories()
    norms = evaluate.load_aoa()
    files = ([result_path(arguments.model, ADAPTATION_DIR)] if arguments.model
             else sorted(ADAPTATION_DIR.glob('*.jsonl')))
    if not files:
        raise SystemExit(f'Nothing collected in {ADAPTATION_DIR}')

    section('Language')
    for path in files:
        replies = read_lines(path)
        if replies.empty:
            continue
        rows = []
        for reply in replies.itertuples():
            text = str(reply.response)
            rows.append({'prompt_id': reply.prompt_id, 'model': reply.model,
                         'replicate': reply.replicate,
                         **measure(text, norms, arguments.floor)})

        model = str(replies['model'].iloc[0])
        written = result_path(model, LANGUAGE_DIR)
        written.unlink(missing_ok=True)
        for row in rows:
            append_line(written, {name: row.get(name, '')
                                  for name in LANGUAGE_COLUMNS})

        measured = sum(1 for row in rows if row[measure_column('FKGL')] != '')
        lengths = sorted(row[measure_column('Response Length')] for row in rows)
        print(f'  {model:<28} {len(rows):>6,} replies, {measured:>6,} long '
              f'enough to measure ({measured / len(rows):>4.0%}), median '
              f'{lengths[len(lengths) // 2]:,} words')

    print(f'\nWritten to {LANGUAGE_DIR}')
    print(f'Replies under {arguments.floor} words carry a length and nothing '
          f'else. Report that share per model: it is not attrition, it is how '
          f'briefly a model refuses.')
