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
                ollama      done_reason length

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
    reasons = [str(body.get('stop_reason') or ''), str(body.get('status') or ''),
               str(body.get('done_reason') or '')]
    for choice in (body.get('choices') or body.get('candidates') or []):
        reasons.append(str(choice.get('finish_reason')
                           or choice.get('finishReason') or ''))
    return [reason.lower() for reason in reasons if reason]


# Words a provider uses when it, rather than the model, stopped the reply.
# Anthropic reports this as an error on the record instead of a stop reason, so
# there is no field to read and the message is what identifies it.
FILTERED = ('content filtering policy', 'content_filter', 'blocked by')

# Stop reasons that mean the provider withheld the reply rather than the model
# finishing it. Google reports these on the candidate, where a reply that was
# blocked after generation has a stop reason and no content at all.
BLOCKING = {'content_filter', 'prohibited_content', 'safety', 'blocklist',
            'spii', 'image_safety', 'recitation'}


# Define function to read the two flags off one raw record. The body is enough
# for most providers, but a reply the provider filtered may have no body at all,
# so the record is taken too: Anthropic returns an error in place of a message,
# and reading only the body would record that as an ordinary empty reply.
def flags_of(body, record=None):
    finishes = finishes_of(body)
    blocked = ((body.get('promptFeedback') or {}).get('blockReason')
               or (body.get('prompt_feedback') or {}).get('block_reason')
               or next((reason.upper() for reason in finishes
                        if reason in BLOCKING), ''))
    if not blocked and record:
        said = str((record.get('result') or {}).get('error')
                   or record.get('error') or '').lower()
        if any(word in said for word in FILTERED):
            blocked = 'CONTENT_FILTER'
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
        records = [json.loads(line) for line in path.read_text().splitlines()
                   if line.strip()]
        # A record the provider refused carries no model, because there is no
        # reply for one to be reported on, and a batch file is named after the
        # job rather than the model. So the file itself is the evidence: it
        # holds one pass, and whichever model its other records name is the one
        # the refused record belongs to. Filtering on the model alone would drop
        # exactly the records this function exists to find.
        named = [model_of(record) for record in records if model_of(record)]
        belongs = max(set(named), key=named.count) if named else ''
        if model and belongs and belongs != model:
            continue
        for record in records:
            if model and not belongs and model_of(record) != model:
                continue
            found[key_of(record)] = flags_of(body_of(record), record)
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
        counts['matched'] += key in known
        if key in known:
            reply['blocked'], reply['truncated'] = known[key]
        else:
            # No raw record, which means the provider file has been moved or
            # cleared rather than that nothing happened. What ingest already
            # wrote is the better answer than a blank, so it is kept: reading
            # this file must never be able to erase what reading it once found.
            reply['blocked'] = str(reply.get('blocked') or '')
            reply['truncated'] = str(reply.get('truncated', '')).lower() in (
                'true', '1')
        blocked, truncated = reply['blocked'], reply['truncated']
        empty = not str(reply.get('response') or '').strip()
        counts['blocked'] += bool(blocked)
        counts['truncated'] += bool(truncated)
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
    if counts['matched'] < counts['replies']:
        print(f"  {counts['replies'] - counts['matched']:,} had no raw record, "
              f"so their flags are whatever ingest wrote. Put the provider "
              f"files back in data/batches to check them.")
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
