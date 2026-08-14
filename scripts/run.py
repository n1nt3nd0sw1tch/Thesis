"""Puts the benchmark to a model, and checks the panel before doing so.

    python scripts/run.py check
    python scripts/run.py check --model qwen3:8b --prompt-id sub-h1-age09

    ollama pull qwen3:8b
    python scripts/run.py generate --model qwen3:8b --backend ollama

    python scripts/run.py generate --backend mlx --limit 60 \
        --model mlx-community/Qwen2.5-7B-Instruct-4bit

Replies append one line at a time and any prompt already collected is skipped,
so a run that stops part way resumes where it left off. That matters because a
full pass is tens of thousands of generations and will not finish in one
sitting. Run it with a limit first: the rate it reports is enough to estimate the
whole pass before committing a night to it. Progress prints once a minute rather
than per reply, so the log stays short enough to read the morning after.

The check looks every identifier up against its provider, so a renamed or
retired model fails here rather than part way through generation, and local
models are checked against the Hugging Face Hub so a missing repository is
caught before a cluster job starts. Naming a model puts one prompt through
generation, scoring and comparison as well; passing --reply instead scores a
supplied text, which exercises the scoring path without loading a large model.
"""

import argparse
import textwrap

import pandas as pd
import requests
from backends import BACKENDS, ask, generate
from evaluate import (JUDGE_TEMPERATURE, JUDGE_TOKENS, OLLAMA_JUDGE, build_item,
                      build_policy, compare, describe, read)
from scripts.settings import (ADAPTATION_DIR, BENCHMARK_PATH, GENERATION, JUDGE,
                      MODELS, PROMPTS_PATH, PROVIDER_KEYS)
from utils import (announce, api_key, collect, make_directories, outstanding,
                   read_all, read_lines, read_table, result_path, section,
                   shape_of)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

TIMEOUT = 30

# Where each provider lists the models a key can reach
LISTINGS = {
    'openai': {'url': 'https://api.openai.com/v1/models',
               'headers': lambda key: {'Authorization': f'Bearer {key}'},
               'path': ('data', 'id')},
    'anthropic': {'url': 'https://api.anthropic.com/v1/models',
                  'headers': lambda key: {'x-api-key': key,
                                          'anthropic-version': '2023-06-01'},
                  'path': ('data', 'id')},
    'google': {'url': 'https://generativelanguage.googleapis.com/v1beta/models',
               'headers': lambda key: {'x-goog-api-key': key},
               'path': ('models', 'name')},
}
HUB = 'https://huggingface.co/api/models'

# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

# Define function to collect a reply to every prompt, once per replicate
def run_generation(arguments):
    section('Generation')
    prompts = read_table(PROMPTS_PATH)
    by_id = dict(zip(prompts['prompt_id'], prompts['prompt']))
    path = result_path(arguments.model, ADAPTATION_DIR)

    wanted = [{'prompt_id': prompt_id, 'model': arguments.model,
               'replicate': replicate, 'backend': arguments.backend,
               'temperature': arguments.temperature}
              for prompt_id in prompts['prompt_id']
              for replicate in range(1, arguments.replicates + 1)]
    pending = outstanding(wanted=wanted, collected=read_lines(path),
                          keys=['prompt_id', 'replicate'])

    print(f'{arguments.model} on {arguments.backend}, {len(prompts)} prompts '
          f'times {arguments.replicates} replicates')
    pending = announce(path=path, wanted=wanted, pending=pending,
                       limit=arguments.limit)
    if not pending:
        raise SystemExit('nothing outstanding')

    def produce(item):
        return {'response': ask(item['backend'], item['model'],
                                by_id[item['prompt_id']], arguments.max_tokens,
                                item['temperature'])}

    failures = collect(pending=pending, produce=produce, path=path)

    section('Collected')
    everything = read_all(ADAPTATION_DIR)
    print(f'{shape_of(everything)} across {ADAPTATION_DIR.name}')
    print(everything.groupby('model').size().to_string())
    return failures


# ----------------------------------------------------------------------------
# The panel
# ----------------------------------------------------------------------------

# Define function to list the models one api key can reach
def list_available(provider, key):
    listing = LISTINGS[provider]
    response = requests.get(listing['url'], timeout=TIMEOUT,
                            headers=listing['headers'](key))
    response.raise_for_status()
    collection, field = listing['path']
    return {entry[field].split('/')[-1]
            for entry in response.json().get(collection, [])}


# Define function to check one api model against its provider
def check_api(spec):
    key = api_key(spec['provider'])
    if not key:
        return f'no {PROVIDER_KEYS[spec["provider"]]} in .env'
    try:
        available = list_available(spec['provider'], key)
    except Exception as error:
        return f'{type(error).__name__}: {error}'
    return 'ok' if spec['id'].split('/')[-1] in available else 'not offered to this key'


# Define function to check one local model against the hub
def check_local(spec):
    try:
        response = requests.get(f'{HUB}/{spec["id"]}', timeout=TIMEOUT)
    except Exception as error:
        return f'{type(error).__name__}: {error}'
    if response.status_code == 200:
        return 'ok'
    return 'gated, needs a licence accepted' if response.status_code == 401 \
        else f'http {response.status_code}'


# Define function to check every model in the panel
def check_panel(models, judge):
    rows = []
    for name, spec in {**models, 'judge': judge}.items():
        verdict = check_api(spec) if spec['access'] == 'api' else check_local(spec)
        rows.append({'model': name, 'provider': spec['provider'],
                     'access': spec['access'], 'weights': spec['weights'],
                     'id': spec['id'], 'status': verdict})
    return pd.DataFrame(rows)


# Define function to report how many calls the panel implies
def report_cost(models, generation, prompts):
    calls = prompts * len(models) * generation['replicates']
    section('Run size')
    print(f'Prompts: {prompts}')
    print(f'Models: {len(models)}, replicates: {generation["replicates"]}')
    print(f'Replies to generate: {calls}')
    print(f'Replies to judge: {calls}')


# ----------------------------------------------------------------------------
# One prompt end to end
# ----------------------------------------------------------------------------

# Define function to load one prompt with the canonical request behind it
def load_prompt(prompt_id, prompts_path, benchmark_path):
    prompts = read_table(prompts_path)
    benchmark = read_table(benchmark_path)
    chosen = prompts[prompts['prompt_id'] == prompt_id] if prompt_id else prompts
    if chosen.empty:
        raise KeyError(f'{prompt_id} is not in {prompts_path.name}')
    row = chosen.iloc[0]
    return row, benchmark.set_index('scenario_id').loc[row['scenario_id']]


# Define function to print a block of text under a heading
def show(title, text, width=76):
    print(f'{title}:')
    for line in textwrap.wrap(str(text), width=width) or ['']:
        print(f'  {line}')


# Define function to put one prompt through generation, scoring and comparison
def trace(arguments):
    row, scenario = load_prompt(prompt_id=arguments.prompt_id,
                                prompts_path=PROMPTS_PATH,
                                benchmark_path=BENCHMARK_PATH)

    section('Prompt')
    print(f'{row["prompt_id"]}  {scenario["domain"]} / '
          f'{scenario["scenario_type"]} / {row["condition"]}')
    show('Sent to the model', row['prompt'])

    section('Reply')
    if arguments.reply:
        print('supplied, not generated')
        reply = arguments.reply
    else:
        print(f'generating with {arguments.model} on {arguments.backend}')
        reply = ask(arguments.backend, arguments.model, row['prompt'])
    show('Returned', reply)

    section('Judgement')
    if arguments.verdict:
        print('supplied, classifier not called')
        output = arguments.verdict
    else:
        print(f'scoring with {arguments.judge}')
        # the classifier sees the canonical request, never the cued variant or
        # the opener, so it cannot infer which condition produced the reply
        output = generate(arguments.backend, arguments.judge,
                          [{'role': 'system', 'content': build_policy()},
                           {'role': 'user',
                            'content': build_item(scenario['request'], reply)}],
                          max_tokens=JUDGE_TOKENS,
                          temperature=JUDGE_TEMPERATURE)
    verdict, problems = read(output)
    for problem in problems:
        print(f'  unreadable, {problem}')
    for field, value in (verdict or {}).items():
        print(f'  {field:<20} {value}')

    section('Comparison')
    expected = row['expected_answer']
    observed = (verdict or {}).get('answer', '')
    deviation = compare(observed=observed, expected=expected)
    print(f'  expected             {expected or "none, this is the control"}')
    print(f'  observed             {observed or "not read"}')
    print(f'  deviation            '
          f'{deviation if deviation is not None else "n/a"}  ({describe(deviation)})')


# Define function to check the panel, and one prompt where a model is named
def run_check(arguments):
    section('Model panel')
    panel = check_panel(models=MODELS, judge=JUDGE)
    print(f'Panel: {shape_of(panel)}')
    print(panel.to_string(index=False))
    failed = panel[panel['status'] != 'ok']
    if len(failed):
        print(f'\n{len(failed)} models are not reachable')

    prompts = read_table(PROMPTS_PATH) if PROMPTS_PATH.exists() else pd.DataFrame()
    report_cost(models=MODELS, generation=GENERATION, prompts=len(prompts))

    # the trace calls a model, so it runs only when one is named or a reply is
    # supplied to score in its place
    if arguments.model or arguments.reply:
        trace(arguments)
    return 0


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['check', 'generate'])
    parser.add_argument('--model', default='')
    parser.add_argument('--backend', default='ollama', choices=list(BACKENDS))
    parser.add_argument('--judge', default='')
    parser.add_argument('--replicates', type=int, default=GENERATION['replicates'])
    parser.add_argument('--max-tokens', type=int, default=GENERATION['max_tokens'])
    parser.add_argument('--temperature', type=float,
                        default=GENERATION['temperature'])
    parser.add_argument('--limit', type=int, default=0,
                        help='stop after this many replies, to time a pass')
    parser.add_argument('--prompt-id', default='', help='the prompt to trace')
    parser.add_argument('--reply', default='',
                        help='skip generation and score this text instead')
    parser.add_argument('--verdict', default='',
                        help='skip the classifier and read this output instead')
    arguments = parser.parse_args()
    if not arguments.judge:
        arguments.judge = (OLLAMA_JUDGE if arguments.backend == 'ollama'
                           else JUDGE['id'])

    make_directories()
    if arguments.stage == 'generate':
        if not arguments.model:
            raise SystemExit('--model is needed to generate')
        failures = run_generation(arguments)
    else:
        failures = run_check(arguments)

    if failures:
        print(f'\n{failures} failed this pass, run again to retry them')
