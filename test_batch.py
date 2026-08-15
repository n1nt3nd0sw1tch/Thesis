"""Puts one prompt through the whole batch route, end to end.

    python test_batch.py                          gpt-5.6-luna, first prompt
    python test_batch.py claude-sonnet-5
    python test_batch.py gpt-5.6-luna bod-a3-age09

Run it from the repository root. It writes a one line request file, submits it,
waits for the job to finish, and prints the reply with what it cost and how many
tokens it used. Nothing is written into results/, so a real pass is untouched.

A batch of one costs a fraction of a penny and usually completes in a couple of
minutes, which is cheap enough to run before every full submission and is the
only thing that proves the payload a provider will actually accept.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, 'scripts')
import backends
import run
import settings
import utils

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'gpt-5.6-luna'
PROMPT_ID = sys.argv[2] if len(sys.argv) > 2 else ''
WAIT = 600          # Seconds to wait before giving up on the job
POLL = 15           # Seconds between checks

# ----------------------------------------------------------------------------
# The one request
# ----------------------------------------------------------------------------

provider = backends.provider_of(MODEL)
if provider != 'openai':
    raise SystemExit(f'{MODEL} is served by {provider}, which submits batches '
                     f'differently. This test covers the OpenAI route only.')
if not utils.api_key(provider):
    raise SystemExit(f'No api key for {provider}. Put it in .env as '
                     f'{settings.PROVIDER_KEYS[provider]}.')

prompts = utils.read_table(settings.PROMPTS_PATH)
row = (prompts[prompts['prompt_id'] == PROMPT_ID] if PROMPT_ID else prompts).iloc[0]
body = backends.build_payload(
    provider, MODEL, [{'role': 'user', 'content': row['prompt']}],
    settings.GENERATION['max_tokens'], settings.GENERATION['temperature'])

path = settings.BATCHES_DIR / 'test_requests.jsonl'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({'custom_id': f'{row["prompt_id"]}-r1', 'method': 'POST',
                            'url': '/v1/responses', 'body': body}) + '\n')

print(f'Model     {MODEL}')
print(f'Prompt    {row["prompt_id"]}  ({row["condition"]})')
print(f'Sending   {row["prompt"]}')
print(f'Payload   {json.dumps(body)}')

# ----------------------------------------------------------------------------
# Submit and wait
# ----------------------------------------------------------------------------

from openai import OpenAI

client = OpenAI(api_key=utils.api_key('openai'))
uploaded = client.files.create(file=open(path, 'rb'), purpose='batch')
job = client.batches.create(input_file_id=uploaded.id, endpoint='/v1/responses',
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
        errors = client.files.content(job.error_file_id).read().decode()
        print(f'\nErrors\n{errors[:600]}')
    raise SystemExit(f'Job ended as {job.status}, so the payload was rejected. '
                     f'Fix it before submitting a full pass.')

# ----------------------------------------------------------------------------
# What came back
# ----------------------------------------------------------------------------

result = json.loads(client.files.content(job.output_file_id).read().decode()
                    .splitlines()[0])
status = result['response']['status_code']
returned = result['response']['body']

if status != 200:
    print(f'\nRejected with {status}')
    print(json.dumps(returned, indent=2)[:600])
    raise SystemExit('The payload was rejected. Fix it before a full pass.')

usage = returned.get('usage', {})
reasoning = (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
price = backends.panel_entry(MODEL, 'price') or {}
cost = (usage.get('input_tokens', 0) * price.get('input', 0)
        + usage.get('output_tokens', 0) * price.get('output', 0)) / 1e6

print(f'\nReply\n{backends.read_reply(provider, returned)}')
print(f'\nTokens    {usage.get("input_tokens")} in, {usage.get("output_tokens")} out, '
      f'{reasoning} of them reasoning')
print(f'Cap       {returned.get("max_output_tokens")}, '
      f'{"hit" if returned.get("incomplete_details") else "not hit"}')
print(f'Decoded   temperature {returned.get("temperature")}, '
      f'top_p {returned.get("top_p")}, '
      f'reasoning {(returned.get("reasoning") or {}).get("effort")}')
print(f'Cost      ${cost:.5f} standard, ${cost / 2:.5f} batched')

calls = len(prompts) * settings.GENERATION['replicates']
print(f'\nA full pass of {calls:,} at this rate: '
      f'${cost * calls:.2f} standard, ${cost * calls / 2:.2f} batched')
