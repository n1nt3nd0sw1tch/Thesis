"""Puts one benchmark prompt to each candidate cloud model, to see which can join.

    python test_cloud_models.py
    python test_cloud_models.py bod-a2 age09
    python test_cloud_models.py bod-a2 age09 gemma4:31b glm-5.2

Run it from the repository root. Reads the prompts, calls each candidate once,
writes nothing.

The panel has five models and room for a sixth. Ollama's cloud serves a dozen
open weight models that could fill it, but a list of names does not say which
your subscription reaches, which will answer without thinking when told not to,
or which returns something the pipeline can read. One call each settles all
three.

Reasoning is switched off, because that is what the panel does. Several of these
are reasoning models, so a candidate that ignores the switch and thinks anyway is
one that cannot be held where the other five are, and that is worth finding out
before it is in the design rather than after.

Models are reached through a local Ollama daemon relaying to the cloud, which is
why the names carry a -cloud suffix. That is the route that authenticates.
"""

import json
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, 'scripts')
import settings
import utils

HOST = 'http://localhost:11434'
TIMEOUT = 300

# Open weight models on Ollama's cloud that would add something to the panel.
# Vendors already in it are left out, as is gpt-oss, which is the classifier and
# cannot also be a subject.
CANDIDATES = [
    'gemma4:31b',          # Google open, pairs with gemini-3.5-flash-lite closed
    'qwen3.5:397b',        # Alibaba, the vendor DashScope would not give access to
    'glm-5.2',             # Z.ai
    'kimi-k3',             # Moonshot
    'minimax-m3',          # MiniMax
    'nemotron-3-super',    # NVIDIA
]

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to put one prompt to one cloud model through the daemon
def ask(model, prompt):
    payload = {'model': model if model.endswith('-cloud') else f'{model}-cloud',
               'messages': [{'role': 'user', 'content': prompt}],
               'stream': False, 'keep_alive': '5m', 'think': False,
               'options': {'temperature': settings.GENERATION['temperature'],
                           'top_p': settings.GENERATION['top_p'],
                           'num_predict': settings.GENERATION['max_tokens']}}
    request = urllib.request.Request(
        f'{HOST}/api/chat', method='POST', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read()), '', time.time() - started
    except urllib.error.HTTPError as problem:
        return None, f'{problem.code}: {problem.read().decode()[:90]}', 0
    except urllib.error.URLError as problem:
        return None, str(problem.reason)[:90], 0


# Define function to read what came back, keeping any thinking out of the reply
def unpack(body):
    message = body.get('message', {}) or {}
    return {'reply': str(message.get('content') or '').strip(),
            'thinking': str(message.get('thinking') or '').strip(),
            'in': body.get('prompt_eval_count', 0),
            'out': body.get('eval_count', 0)}


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    arguments = sys.argv[1:]
    scenario = arguments[0] if arguments and '-' in arguments[0] else ''
    condition = arguments[1] if len(arguments) > 1 else 'age09'
    named = [a for a in arguments[2:]] or CANDIDATES

    prompts = utils.read_table(settings.PROMPTS_PATH)
    benchmark = utils.read_table(settings.BENCHMARK_PATH)
    frame = prompts.merge(benchmark[['scenario_id', 'domain', 'scenario_type']],
                          on='scenario_id')
    wanted = frame[frame['condition'] == condition]
    wanted = (wanted[wanted['scenario_id'] == scenario] if scenario
              else wanted[wanted['scenario_type'] == 'Harmful'])
    if wanted.empty:
        raise SystemExit(f'No prompt for {scenario or "a Harmful scenario"} at '
                         f'{condition}')
    row = wanted.iloc[0]

    print(f'Prompt     {row.scenario_id}  {row.domain}, {row.scenario_type}, '
          f'{condition}')
    print(f'Sending    {row.prompt}')
    print(f'Settings   temperature {settings.GENERATION["temperature"]}, cap '
          f'{settings.GENERATION["max_tokens"]}, thinking off')
    print(f'\n{len(named)} candidates through {HOST}\n')

    usable = []
    for model in named:
        body, problem, took = ask(model, row.prompt)
        if body is None:
            print(f'  {model:<20} unreachable: {problem}')
            continue
        seen = unpack(body)
        thought = (f', {len(seen["thinking"].split())} words of thinking despite '
                   f'the switch' if seen['thinking'] else '')
        if not seen['reply']:
            print(f'  {model:<20} answered nothing{thought}')
            continue
        usable.append(model)
        print(f'  {model:<20} {took:>5.1f}s  {seen["in"]} in, {seen["out"]} out'
              f'{thought}')
        for line in [seen['reply'][i:i + 68] for i in range(0, 200, 68)]:
            if line.strip():
                print(f'      {line}')
        print()

    print(f'{len(usable)} of {len(named)} usable: {", ".join(usable) or "none"}')
    print('\nA candidate that thinks anyway cannot be held where the other five')
    print('are. One that answers nothing, or is unreachable, is not a candidate.')
    print('Of the rest, gemma4 is the one that adds a question rather than a row:')
    print('Google open against Google closed, which nothing else in the panel does.')
