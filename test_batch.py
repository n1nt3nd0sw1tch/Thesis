"""Puts one prompt through the whole batch route, end to end.

    python test_batch.py                                  gpt-5.6-luna
    python test_batch.py claude-haiku-4-5-20251001
    python test_batch.py gpt-5.6-luna bod-a3-age09        a chosen prompt

Run it from the repository root. It writes a one line request file, submits it,
waits for the job to finish, and prints the reply with what it cost and how many
tokens it used. Nothing is written into results/, so a real pass is untouched.

A batch of one costs a fraction of a penny and is the only thing that proves the
payload a provider will actually accept. It is worth running before every full
submission, and after any change to the panel or the generation settings.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, 'scripts')
import backends
import settings
import utils

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'gpt-5.6-luna'
PROMPT_ID = sys.argv[2] if len(sys.argv) > 2 else ''
WAIT = 900          # Seconds to wait before giving up on the job
POLL = 15           # Seconds between checks

# ----------------------------------------------------------------------------
# The one request
# ----------------------------------------------------------------------------

provider = backends.provider_of(MODEL)
if provider not in ('openai', 'anthropic'):
    raise SystemExit(f'{MODEL} is served by {provider}, which has no batch route '
                     f'here yet. Use run.py generate --backend api instead.')
if not utils.api_key(provider):
    raise SystemExit(f'No api key for {provider}. Put it in .env as '
                     f'{settings.PROVIDER_KEYS[provider]}.')

prompts = utils.read_table(settings.PROMPTS_PATH)
row = (prompts[prompts['prompt_id'] == PROMPT_ID] if PROMPT_ID else prompts).iloc[0]
body = backends.build_payload(
    provider, MODEL, [{'role': 'user', 'content': row['prompt']}],
    settings.GENERATION['max_tokens'], settings.GENERATION['temperature'])
custom_id = f'{row["prompt_id"]}-r1'

path = settings.BATCHES_DIR / 'test_requests.jsonl'
path.parent.mkdir(parents=True, exist_ok=True)
line = ({'custom_id': custom_id, 'params': body} if provider == 'anthropic'
        else {'custom_id': custom_id, 'method': 'POST', 'url': '/v1/responses',
              'body': body})
path.write_text(json.dumps(line) + '\n')

print(f'Model     {MODEL} on {provider}')
print(f'Prompt    {row["prompt_id"]}  ({row["condition"]})')
print(f'Sending   {row["prompt"]}')
print(f'Payload   {json.dumps(body)}')

# ----------------------------------------------------------------------------
# Submit, wait, and read the one result back
# ----------------------------------------------------------------------------

if provider == 'openai':
    from openai import OpenAI

    client = OpenAI(api_key=utils.api_key('openai'))
    uploaded = client.files.create(file=open(path, 'rb'), purpose='batch')
    job = client.batches.create(input_file_id=uploaded.id,
                                endpoint='/v1/responses',
                                completion_window='24h')
    print(f'\nSubmitted {job.id}')

    started = time.time()
    while time.time() - started < WAIT:
        job = client.batches.retrieve(job.id)
        if job.status in ('completed', 'failed', 'cancelled', 'expired'):
            break
        print(f'  {job.status}, {int(time.time() - started)}s')
        time.sleep(POLL)
    print(f'Finished  {job.status} after {int(time.time() - started)}s')

    if job.status != 'completed':
        if job.error_file_id:
            print(f'\nErrors\n'
                  f'{client.files.content(job.error_file_id).read().decode()[:600]}')
        raise SystemExit(f'Job ended as {job.status}, so the payload was rejected.')

    result = json.loads(client.files.content(job.output_file_id).read()
                        .decode().splitlines()[0])
    if result['response']['status_code'] != 200:
        print(f'\nRejected\n{json.dumps(result["response"]["body"], indent=2)[:600]}')
        raise SystemExit('The payload was rejected. Fix it before a full pass.')
    returned = result['response']['body']
    usage = returned.get('usage', {})
    sent, received = usage.get('input_tokens', 0), usage.get('output_tokens', 0)
    reasoning = (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
    cap_hit = bool(returned.get('incomplete_details'))
    decoded = (f'temperature {returned.get("temperature")}, '
               f'top_p {returned.get("top_p")}, '
               f'reasoning {(returned.get("reasoning") or {}).get("effort")}')

else:
    from anthropic import Anthropic

    client = Anthropic(api_key=utils.api_key('anthropic'))
    job = client.messages.batches.create(requests=[line])
    print(f'\nSubmitted {job.id}')

    started = time.time()
    while time.time() - started < WAIT:
        job = client.messages.batches.retrieve(job.id)
        if job.processing_status == 'ended':
            break
        print(f'  {job.processing_status}, {int(time.time() - started)}s')
        time.sleep(POLL)
    print(f'Finished  {job.processing_status} after {int(time.time() - started)}s')

    if job.processing_status != 'ended':
        raise SystemExit(f'Job is still {job.processing_status} after {WAIT}s.')

    entry = next(iter(client.messages.batches.results(job.id))).model_dump(mode='json')
    outcome = entry.get('result') or {}
    if outcome.get('type') != 'succeeded':
        print(f'\nRejected\n{json.dumps(outcome, indent=2)[:600]}')
        raise SystemExit('The payload was rejected. Fix it before a full pass.')
    returned = outcome['message']
    usage = returned.get('usage', {})
    sent, received = usage.get('input_tokens', 0), usage.get('output_tokens', 0)
    reasoning = 0
    cap_hit = returned.get('stop_reason') == 'max_tokens'
    decoded = (f'stop reason {returned.get("stop_reason")}, '
               f'thinking not requested')

# ----------------------------------------------------------------------------
# What it means for a full pass
# ----------------------------------------------------------------------------

price = backends.panel_entry(MODEL, 'price') or {}
cost = (sent * price.get('input', 0) + received * price.get('output', 0)) / 1e6

print(f'\nReply\n{backends.read_reply(provider, returned)}')
print(f'\nTokens    {sent} in, {received} out'
      + (f', {reasoning} of them reasoning' if reasoning else ''))
print(f'Cap       {settings.GENERATION["max_tokens"]}, '
      f'{"HIT, the reply is cut off" if cap_hit else "not hit"}')
print(f'Decoded   {decoded}')
print(f'Cost      ${cost:.5f} standard, ${cost / 2:.5f} batched')

calls = len(prompts) * settings.GENERATION['replicates']
print(f'\nA full pass of {calls:,} at this rate: '
      f'${cost * calls:.2f} standard, ${cost * calls / 2:.2f} batched')