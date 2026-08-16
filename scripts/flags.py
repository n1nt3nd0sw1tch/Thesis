"""Says whether a request was blocked or a reply was cut off at the token cap.

Each provider reports both in its own words, so the reading of them lives here
rather than in the stage that happens to need it. Ingest calls this as replies
arrive, and `apply` calls it over a file already collected, so the two cannot
drift apart and a pass collected before a flag existed can be brought up to date
without regenerating anything.

    python scripts/flags.py                 report every model, write nothing
    python scripts/flags.py --write         bring results/adaptation/ up to date

    blocked     the request was refused before the model saw it
                google      promptFeedback.blockReason
                openai      finish_reason or type content_filter
                deepseek    finish_reason content_filter
                anthropic   no equivalent, so this is always empty

    truncated   generation stopped at the cap rather than finishing
                openai      incomplete_details set, or status incomplete
                anthropic   stop_reason max_tokens
                google      finishReason MAX_TOKENS
                deepseek    finish_reason length

An empty reply is then explained rather than merely counted. It is either a
blocked prompt, which produced nothing, or a truncated one where the model spent
the whole cap before writing any answer.
"""

import json
import sys

from settings import ADAPTATION_DIR, BATCHES_DIR, RESPONSE_COLUMNS
from utils import read_lines, result_path, section

# What each provider calls the cap being reached
CAP_REACHED = {'max_tokens', 'length', 'incomplete'}

# ----------------------------------------------------------------------------
# Reading one record
# ----------------------------------------------------------------------------

# Define function to find the body of one raw record, whichever provider wrote it
def body_of(record):
    if 'result' in record:                                    # anthropic
        return (record['result'] or {}).get('message') or {}
    response = record.get('response') or {}
    return response.get('body') or response                   # openai, google


# Define function to read the model a raw record came from, so that a file
# identifies itself rather than being identified by its name
def model_of(record):
    body = body_of(record)
    return body.get('model') or body.get('modelVersion') or ''


# Define function to read the identifier a raw record was asked under
def key_of(record):
    key = str(record.get('custom_id') or record.get('key') or '')
    prompt_id, _, replicate = key.rpartition('-r')
    return prompt_id, replicate


# Define function to say why generation stopped, in whichever field the provider
# put it
def finishes_of(body):
    reasons = [str(body.get('stop_reason') or ''), str(body.get('status') or '')]
    for choice in (body.get('choices') or body.get('candidates') or []):
        reasons.append(str(choice.get('finish_reason')
                           or choice.get('finishReason') or ''))
    return [reason.lower() for reason in reasons if reason]


# Define function to read the two flags off one reply body
def flags_of(body):
    finishes = finishes_of(body)
    blocked = ((body.get('promptFeedback') or {}).get('blockReason')
               or (body.get('prompt_feedback') or {}).get('block_reason')
               or ('CONTENT_FILTER' if 'content_filter' in finishes else ''))
    truncated = bool(body.get('incomplete_details')) or bool(
        CAP_REACHED & set(finishes))
    return str(blocked), truncated


# ----------------------------------------------------------------------------
# Over a whole pass
# ----------------------------------------------------------------------------

# Define function to read the flags of every raw record on disk, keyed by the
# prompt and replicate they were asked under
def raw_flags(model=''):
    found = {}
    for path in sorted(BATCHES_DIR.glob('*output.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if model and model_of(record) != model:
                continue
            found[key_of(record)] = flags_of(body_of(record))
    return found


# Define function to bring one model's collected replies up to date with what
# the raw records say, and report what that leaves
def apply(model, write=True):
    path = result_path(model, ADAPTATION_DIR)
    replies = read_lines(path)
    if replies.empty:
        return {'replies': 0, 'matched': 0, 'blocked': 0, 'truncated': 0,
                'empty': 0, 'unexplained': 0}

    known = raw_flags(model)
    rows, counts = [], {'replies': len(replies), 'matched': 0, 'blocked': 0,
                        'truncated': 0, 'empty': 0, 'unexplained': 0}
    for reply in replies.to_dict('records'):
        key = (str(reply['prompt_id']), str(reply['replicate']))
        blocked, truncated = known.get(key, ('', False))
        counts['matched'] += key in known
        reply['blocked'], reply['truncated'] = blocked, truncated
        empty = not str(reply.get('response') or '').strip()
        counts['blocked'] += bool(blocked)
        counts['truncated'] += truncated
        counts['empty'] += empty
        counts['unexplained'] += empty and not blocked and not truncated
        rows.append({name: reply.get(name, '') for name in RESPONSE_COLUMNS})

    if write:
        path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    return counts


# Define function to report what one model's flags amount to
def report(model, counts):
    print(f'{model}')
    print(f"  {counts['replies']:,} replies, {counts['matched']:,} matched to a "
          f"raw record")
    print(f"  {counts['blocked']} blocked, {counts['truncated']} truncated, "
          f"{counts['empty']} empty")
    if counts['unexplained']:
        print(f"  {counts['unexplained']} empty for neither reason, worth reading")


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    write = '--write' in sys.argv
    models = sorted({model_of(json.loads(line))
                     for path in BATCHES_DIR.glob('*output.jsonl')
                     for line in path.read_text().splitlines()[:1]} - {''})
    if not models:
        raise SystemExit(f'No raw output files in {BATCHES_DIR}')

    section('Flags')
    for model in models:
        report(model, apply(model, write=write))
    if not write:
        print('\nNothing written. Run again with --write to update '
              'results/adaptation/.')
