"""Puts one prompt through the whole route a model uses, end to end.

    python test_batch.py                                  gpt-5.6-luna
    python test_batch.py claude-haiku-4-5-20251001
    python test_batch.py deepseek-v4-flash                live, no batch queue
    python test_batch.py gpt-5.6-luna bod-a3-age09        a chosen prompt

Run it from the repository root. It writes a one line request file, submits it,
waits for the job to finish, and prints the reply with what it cost and how many
tokens it used. Nothing is written into results/, so a real pass is untouched.

A batch of one costs a fraction of a penny and is the only thing that proves the
payload a provider will actually accept. It is worth running before every full
submission, and after any change to the panel or the generation settings.
"""
# test
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
WAIT = 1800         # Seconds to wait before giving up on the job
POLL = 10           # Seconds before the first check
BACKOFF = 1.5       # How much longer to wait before each check after that
POLL_MAX = 120      # Longest gap between checks



# Define function to wait for a job, checking less often as it goes on. Each
# check is itself an api request and appears in the provider's usage, so a fixed
# short interval turns a quiet ten minute wait into forty logged requests.
def wait_for(describe, finished):
    started, gap = time.time(), POLL
    while time.time() - started < WAIT:
        state = describe()
        if finished(state):
            return state, int(time.time() - started), True
        print(f'  {state}, {int(time.time() - started)}s')
        time.sleep(gap)
        gap = min(gap * BACKOFF, POLL_MAX)
    return describe(), int(time.time() - started), False


# Define function to import the Mistral client, which moved between SDK versions
# and reports an unhelpful error when the package is shadowed by a bare
# directory of the same name
def mistral_client(key):
    try:
        from mistralai import Mistral
    except ImportError:
        try:
            from mistralai.client import Mistral
        except ImportError as problem:
            import mistralai
            where = getattr(mistralai, '__file__', None)
            raise SystemExit(
                f'Could not import the Mistral client: {problem}\n'
                + (f'mistralai resolves to {where}\n' if where else
                   'mistralai resolved to a directory with no package in it, '
                   'so either\nit is not installed or something of that name is '
                   'shadowing it.\n')
                + 'Try: pip install mistralai')
    return Mistral(api_key=key)


# ----------------------------------------------------------------------------
# The one request
# ----------------------------------------------------------------------------

provider = backends.provider_of(MODEL)
if provider not in ('openai', 'anthropic', 'google', 'deepseek', 'mistral'):
    raise SystemExit(f'{MODEL} is served by {provider}, which this script does '
                     f'not cover. Use run.py generate --backend api instead.')
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
if provider == 'deepseek':
    line = {'custom_id': custom_id, 'body': body}
elif provider == 'mistral':
    # the model is named on the job, not on the line
    line = {'custom_id': custom_id,
            'body': {k: v for k, v in body.items() if k != 'model'}}
elif provider == 'anthropic':
    line = {'custom_id': custom_id, 'params': body}
elif provider == 'google':
    line = {'key': custom_id, 'request': body}
else:
    line = {'custom_id': custom_id, 'method': 'POST', 'url': '/v1/responses',
            'body': body}
path.write_text(json.dumps(line) + '\n')

print(f'Model     {MODEL} on {provider}')
print(f'Prompt    {row["prompt_id"]}  ({row["condition"]})')
print(f'Sending   {row["prompt"]}')
print(f'Payload   {json.dumps(body)}')

# ----------------------------------------------------------------------------
# Submit, wait, and read the one result back
# ----------------------------------------------------------------------------

if provider == 'deepseek':
    # No batch queue here, so the one request goes straight out. The point of
    # the test is the same: prove the payload is accepted and measure what a
    # reply costs before committing to four thousand of them.
    print('\nNo batch queue for this provider, sending live')
    reply = backends.generate_api(MODEL, [{'role': 'user', 'content': row['prompt']}],
                                  settings.GENERATION['max_tokens'],
                                  settings.GENERATION['temperature'])
    sent, received = backends.USAGE['input'], backends.USAGE['output']
    reasoning = backends.LAST_REASONING
    cap_hit = backends.LAST_FINISH == 'length'
    # report what was sent, not what the design asked for: a model that refuses
    # sampling parameters decodes at its own settings whatever the config says
    decoded = (f'temperature {body["temperature"]}, top_p {body["top_p"]}'
               if 'temperature' in body
               else 'provider defaults, this model does not accept temperature '
                    'or top_p')
    returned = None

elif provider == 'openai':
    from openai import OpenAI

    client = OpenAI(api_key=utils.api_key('openai'))
    uploaded = client.files.create(file=open(path, 'rb'), purpose='batch')
    job = client.batches.create(input_file_id=uploaded.id,
                                endpoint='/v1/responses',
                                completion_window='24h')
    print(f'\nSubmitted {job.id}')

    job_id = job.id
    _, waited, _ = wait_for(
        lambda: client.batches.retrieve(job_id).status,
        lambda s: s in ('completed', 'failed', 'cancelled', 'expired'))
    job = client.batches.retrieve(job_id)
    print(f'Finished  {job.status} after {waited}s')

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

elif provider == 'anthropic':
    from anthropic import Anthropic

    client = Anthropic(api_key=utils.api_key('anthropic'))
    job = client.messages.batches.create(requests=[line])
    print(f'\nSubmitted {job.id}')

    job_id = job.id
    _, waited, _ = wait_for(
        lambda: client.messages.batches.retrieve(job_id).processing_status,
        lambda s: s == 'ended')
    job = client.messages.batches.retrieve(job_id)
    print(f'Finished  {job.processing_status} after {waited}s')

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

elif provider == 'mistral':
    client = mistral_client(utils.api_key('mistral'))
    uploaded = client.files.upload(
        file={'file_name': path.name, 'content': path.open('rb')}, purpose='batch')
    job = client.batch.jobs.create(input_files=[uploaded.id], model=MODEL,
                                   endpoint='/v1/chat/completions',
                                   metadata={'job_type': 'one request test'})
    print(f'\nSubmitted {job.id}')

    job_id = job.id
    _, waited, _ = wait_for(
        lambda: client.batch.jobs.get(job_id=job_id).status,
        lambda s: s in ('SUCCESS', 'FAILED', 'TIMEOUT_EXCEEDED', 'CANCELLED'))
    job = client.batch.jobs.get(job_id=job_id)
    print(f'Finished  {job.status} after {waited}s')

    if job.status != 'SUCCESS':
        for problem in (job.errors or []):
            print(f'  {getattr(problem, "message", problem)}')
        if job.error_file:
            print(f'  Rejections in file {job.error_file}')
        raise SystemExit(f'Job ended as {job.status}, so the payload was refused.')

    result = json.loads(client.files.download(file_id=job.output_file)
                        .read().decode().splitlines()[0])
    returned = (result.get('response') or {}).get('body') or {}
    if not returned.get('choices'):
        print(f'\nNo reply\n{json.dumps(result, indent=2)[:600]}')
        raise SystemExit('The payload was refused. Fix it before a full pass.')
    usage = returned.get('usage', {})
    sent, received = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
    reasoning = 0
    finish = (returned['choices'][0] or {}).get('finish_reason', '')
    cap_hit = finish == 'length'
    decoded = f'finish reason {finish}'

else:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=utils.api_key('google'))
    uploaded = client.files.upload(
        file=str(path),
        config=types.UploadFileConfig(display_name='test', mime_type='jsonl'))
    job = client.batches.create(model=MODEL, src=uploaded.name,
                                config={'display_name': 'one-request-test'})
    print(f'\nSubmitted {job.name}')

    job_name = job.name
    _, waited, _ = wait_for(
        lambda: client.batches.get(name=job_name).state.name,
        lambda s: s in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED',
                        'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'))
    job = client.batches.get(name=job_name)
    print(f'Finished  {job.state.name} after {waited}s')

    if job.state.name != 'JOB_STATE_SUCCEEDED':
        raise SystemExit(f'Job ended as {job.state.name}: '
                         f'{getattr(job, "error", "no detail given")}')

    result = json.loads(client.files.download(file=job.dest.file_name)
                        .decode().splitlines()[0])
    if result.get('error'):
        print(f'\nRejected\n{json.dumps(result["error"], indent=2)[:600]}')
        raise SystemExit('The payload was rejected. Fix it before a full pass.')
    returned = result['response']
    usage = returned.get('usageMetadata', {})
    sent = usage.get('promptTokenCount', 0)
    reasoning = usage.get('thoughtsTokenCount', 0)
    received = usage.get('candidatesTokenCount', 0) + reasoning
    finish = (returned.get('candidates') or [{}])[0].get('finishReason', '')
    cap_hit = str(finish).upper() == 'MAX_TOKENS'
    decoded = f'finish reason {finish}, thinking billed as output'

# ----------------------------------------------------------------------------
# What it means for a full pass
# ----------------------------------------------------------------------------

price = backends.panel_entry(MODEL, 'price') or {}
cost = (sent * price.get('input', 0) + received * price.get('output', 0)) / 1e6

print(f'\nReply\n{reply if returned is None else backends.read_reply(provider, returned)}')
print(f'\nTokens    {sent} in, {received} out'
      + (f', {reasoning} of them reasoning' if reasoning else ''))
print(f'Cap       {settings.GENERATION["max_tokens"]}, '
      f'{"HIT, the reply is cut off" if cap_hit else "not hit"}')
print(f'Decoded   {decoded}')
print(f'Cost      ${cost:.5f} standard, ${cost / 2:.5f} batched')

calls = len(prompts) * settings.GENERATION['replicates']
print(f'\nA full pass of {calls:,} at this rate: '
      f'${cost * calls:.2f} standard, ${cost * calls / 2:.2f} batched')