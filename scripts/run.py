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
import json
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import backends
from backends import (BATCH_SIZE, BATCHED, BACKENDS, USAGE, ask, build_payload,
                      call_api,
                      generate, generate_many, provider_of, read_reply,
                      record_usage, spent)
from evaluate import (JUDGE_TEMPERATURE, JUDGE_TOKENS, OLLAMA_JUDGE, build_item,
                      build_policy, compare, describe, read)
from flags import flags_of
from settings import (ADAPTATION_DIR, BENCHMARK_PATH, GENERATION, JUDGE,
                      BATCHES_DIR, MODELS, PROMPTS_PATH, PROVIDER_KEYS)
from utils import (WORKERS, announce, api_key, append_line, collect,
                   make_directories, model_slug, outstanding, read_all,
                   read_lines, read_table, result_path, section, shape_of)

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
        raise SystemExit('Nothing outstanding')

    def produce(item):
        return {'response': ask(item['backend'], item['model'],
                                by_id[item['prompt_id']], arguments.max_tokens,
                                item['temperature'])}

    def produce_batch(group):
        replies = generate_many(
            arguments.backend, arguments.model,
            [[{'role': 'user', 'content': by_id[item['prompt_id']]}]
             for item in group],
            arguments.max_tokens, arguments.temperature)
        return [{'response': reply} for reply in replies]

    def meter():
        return (backends.spent(arguments.model),
                backends.USAGE['input'] + backends.USAGE['output'])

    batched = arguments.backend in BATCHED
    if batched:
        print(f'{arguments.batch_size} conversations to the GPU at a time')
    elif arguments.workers > 1:
        print(f'{arguments.workers} calls in flight at once')
    failures = collect(pending=pending, produce=produce, path=path,
                       label=arguments.model,
                       meter=meter if arguments.backend == 'api' else None,
                       produce_batch=produce_batch if batched else None,
                       batch_size=arguments.batch_size,
                       workers=arguments.workers)

    section('Collected')
    everything = read_all(ADAPTATION_DIR)
    print(f'{shape_of(everything)} across {ADAPTATION_DIR.name}')
    print(everything.groupby('model').size().to_string())
    spent = backends.spent(arguments.model)
    if spent is not None:
        usage = backends.USAGE
        print(f'\n{usage["calls"]:,} api calls this pass, '
              f'{usage["input"]:,} input and {usage["output"]:,} output tokens, '
              f'${spent:.2f}')
    return failures




# ----------------------------------------------------------------------------
# Live runs, in parts
# ----------------------------------------------------------------------------

# A provider with no batch queue is generated live. The pass is cut into parts
# so that a long run is checkpointed rather than all or nothing, and so that
# progress is visible in whole chunks rather than only as a rate. Each part
# writes the raw responses in the shape a batch job would have returned, which
# means the same read_batch ingests them and the same evidence is left behind.

# Define function to name the file one part of a live pass writes to
def part_path(model, part):
    return BATCHES_DIR / f'part{part}-{model_slug(model)}_output.jsonl'


# Define function to name the file the parts are joined into
def joined_path(model):
    return BATCHES_DIR / f'live-{model_slug(model)}_output.jsonl'


# Define function to list what one part of a live pass should ask for. The split
# is over what is outstanding at the moment it is taken, so a part rerun after a
# failure asks only for what that part still lacks.
def part_items(model, part, parts, replicates=None, temperature=None):
    replicates = GENERATION['replicates'] if replicates is None else replicates
    temperature = GENERATION['temperature'] if temperature is None else temperature
    prompts = read_table(PROMPTS_PATH)
    wanted = [{'prompt_id': prompt_id, 'model': model, 'replicate': replicate,
               'backend': 'api', 'temperature': temperature}
              for prompt_id in prompts['prompt_id']
              for replicate in range(1, replicates + 1)]
    pending = outstanding(wanted=wanted, keys=['prompt_id', 'replicate'],
                          collected=read_lines(result_path(model, ADAPTATION_DIR)))
    size = -(-len(wanted) // parts)          # ceiling, so the last part is short
    lower, upper = (part - 1) * size, part * size
    order = {(item['prompt_id'], str(item['replicate'])): i
             for i, item in enumerate(wanted)}
    return [item for item in pending
            if lower <= order[(item['prompt_id'], str(item['replicate']))] < upper]


# Define function to generate one part of a live pass, writing each raw response
# as it arrives so that an interrupted part loses only the call in flight
def generate_part(model, part, parts, replicates=None, max_tokens=None,
                  temperature=None, workers=None):
    max_tokens = GENERATION['max_tokens'] if max_tokens is None else max_tokens
    items = part_items(model, part, parts, replicates, temperature)
    path = part_path(model, part)
    if not items:
        return path, 0, 0

    prompts = read_table(PROMPTS_PATH)
    by_id = dict(zip(prompts['prompt_id'], prompts['prompt']))

    def produce(item):
        body = call_api(model, [{'role': 'user', 'content': by_id[item['prompt_id']]}],
                        max_tokens, item['temperature'])
        append_line(path, {'custom_id': f'{item["prompt_id"]}-r{item["replicate"]}',
                           'response': {'status_code': 200, 'body': body}})
        return {}

    failures = collect(pending=items, produce=produce,
                       path=path.with_suffix('.log.jsonl'),
                       label=f'{model} part {part}',
                       meter=lambda: (spent(model),
                                      USAGE['input'] + USAGE['output']),
                       workers=WORKERS if workers is None else workers)
    path.with_suffix('.log.jsonl').unlink(missing_ok=True)
    return path, len(items), failures


# Define function to join the parts into one file and remove them, so that a
# live pass leaves a single record of what the provider returned
def join_parts(model, parts):
    joined = joined_path(model)
    present = [part_path(model, part) for part in range(1, parts + 1)]
    present = [path for path in present if path.exists()]
    # Nothing to join means the parts were joined already. Opening the target
    # for writing here would empty the file that holds the whole pass, so the
    # earlier join is reported instead.
    if not present:
        already = (len(joined.read_text().splitlines()) if joined.exists() else 0)
        return joined, already

    lines = [line for path in present
             for line in path.read_text().splitlines() if line.strip()]
    joined.write_text('\n'.join(lines) + '\n')
    for path in present:
        path.unlink()
    return joined, len(lines)


# ----------------------------------------------------------------------------
# The provider batch queues
# ----------------------------------------------------------------------------

# A batch job halves the price and runs asynchronously, so it is submitted from
# the provider's console rather than driven from here. Two stages bracket it:
# export writes the file to upload, ingest reads the file that comes back. The
# body of each request is built by the same function the live path uses, so the
# two cannot drift apart, and ingest writes the same records live generation
# does, so nothing downstream knows or cares which route a reply took.

# Define function to name the file a batch is written to or read from. Before a
# job exists the file is named after the model; once the provider has given an
# identifier it is renamed after that, matching the name the results come back
# under, so the request and reply files for a job sit together.
def batch_path(model_id, suffix, job_id=""):
    stem = job_id if job_id else f"pending-{model_slug(model_id)}"
    return BATCHES_DIR / f"{stem}_{suffix}.jsonl"


# Define function to rename a written request file after the job it became
def name_after_job(model_id, job_id):
    pending = batch_path(model_id, "requests")
    named = batch_path(model_id, "requests", job_id)
    if pending.exists():
        pending.replace(named)
    return named


# Define function to write the batch file to upload, skipping anything already
# collected so that it composes with a run that stopped part way
def write_batch(model, endpoint='/v1/responses', replicates=None, max_tokens=None,
                temperature=None, limit=0, fresh=False):
    replicates = GENERATION['replicates'] if replicates is None else replicates
    max_tokens = GENERATION['max_tokens'] if max_tokens is None else max_tokens
    temperature = GENERATION['temperature'] if temperature is None else temperature
    prompts = read_table(PROMPTS_PATH)
    provider = provider_of(model)

    wanted = [{'prompt_id': prompt_id, 'model': model, 'replicate': replicate,
               'backend': 'api', 'temperature': temperature}
              for prompt_id in prompts['prompt_id']
              for replicate in range(1, replicates + 1)]
    # a rerun asks the same prompts again, so what is already collected is not a
    # reason to skip them
    pending = wanted if fresh else outstanding(
        wanted=wanted, keys=['prompt_id', 'replicate'],
        collected=read_lines(result_path(model, ADAPTATION_DIR)))
    if limit:
        pending = pending[:limit]
    if not pending:
        return None, 0

    by_id = dict(zip(prompts['prompt_id'], prompts['prompt']))
    path = batch_path(model, 'requests')
    path.unlink(missing_ok=True)
    for item in pending:
        body = build_payload(
            provider, model,
            [{'role': 'user', 'content': by_id[item['prompt_id']]}],
            max_tokens, item['temperature'])
        custom_id = f'{item["prompt_id"]}-r{item["replicate"]}'
        # Each provider names the parts differently. OpenAI takes a file of
        # addressed requests, Anthropic the parameters alone with the endpoint
        # fixed for the job, and Google a key beside a bare request, with the
        # model named once when the job is created rather than per line.
        if provider == 'anthropic':
            line = {'custom_id': custom_id, 'params': body}
        elif provider == 'google':
            line = {'key': custom_id, 'request': body}
        else:
            line = {'custom_id': custom_id, 'method': 'POST', 'url': endpoint,
                    'body': body}
        append_line(path, line)
    return path, len(pending)


# Define function to set a model's collected replies aside before it is run
# again. Ingest appends and skips what it already has, so a rerun over an
# existing file would either be skipped entirely or leave two replies per prompt
# collected under different request parameters. The earlier pass is kept rather
# than deleted, outside the directory the pipeline reads, because it is evidence
# of what the model did under the conditions that produced it.
def set_aside_replies(model):
    current = result_path(model, ADAPTATION_DIR)
    if not current.exists():
        return None
    aside = ADAPTATION_DIR.parent / 'superseded'
    aside.mkdir(parents=True, exist_ok=True)
    moved = aside / f'{current.stem}.{datetime.now():%Y%m%d-%H%M}.jsonl'
    current.replace(moved)
    return moved


# Define function to read a finished batch into the results, writing the same
# records live generation writes so that nothing downstream can tell them apart.
# Anything already collected for a prompt and replicate is skipped: ingesting the
# same file twice, or two batches covering the same prompts, would otherwise
# leave duplicate rows that no later stage could tell apart.
def read_batch(model, source, temperature=None):
    temperature = GENERATION['temperature'] if temperature is None else temperature
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f'{source} not found')
    provider = provider_of(model)
    path = result_path(model, ADAPTATION_DIR)

    collected = read_lines(path)
    seen = set() if collected.empty else {
        (str(row.prompt_id), str(row.replicate)) for row in collected.itertuples()}

    read, failed, truncated, repeated, blocked = 0, 0, 0, 0, 0
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # Google returns the identifier as key, the other two as custom_id
        prompt_id, _, replicate = str(row.get('custom_id')
                                      or row.get('key')).rpartition('-r')
        if (prompt_id, replicate) in seen:
            repeated += 1
            continue

        # Anthropic reports the outcome under result; OpenAI under response
        # with a status code; Google under response with an error beside it
        if 'key' in row and 'result' not in row:
            body = row.get('response') or {}
            failure = row.get('error') or (None if body else row)
        elif 'result' in row:
            outcome = row['result'] or {}
            body = outcome.get('message') or {}
            failure = None if outcome.get('type') == 'succeeded' else outcome
        else:
            response = row.get('response') or {}
            body = response.get('body') or {}
            failure = (row.get('error')
                       or (body if response.get('status_code', 200) != 200 else None))
        # A prompt refused by the provider's own filter never reached the model,
        # so there is no reply to score. It is recorded rather than dropped,
        # because where a provider intervenes before generation is itself a
        # result, and it is kept out of the error field so that a rerun does not
        # keep resubmitting something that will be blocked again.
        stopped, cut = flags_of(body)
        error = ''
        if failure:
            error = str(failure)[:200]
            failed += 1
        elif stopped:
            blocked += 1
        else:
            record_usage(provider, body)
            # a reply stopped by the token cap has a censored length rather than
            # a measured one, and Response Length is an outcome measure
            truncated += cut

        append_line(path, {'prompt_id': prompt_id, 'model': model,
                           'replicate': replicate, 'backend': 'batch',
                           'temperature': temperature, 'error': error,
                           'blocked': stopped, 'truncated': cut,
                           'response': '' if error or stopped
                                       else read_reply(provider, body)})
        seen.add((prompt_id, replicate))
        read += 1
    return read, failed, truncated, repeated, blocked


# Define function to write the batch file from the command line
def run_export(arguments):
    section('Batch export')
    path, count = write_batch(model=arguments.model, endpoint=arguments.endpoint,
                              replicates=arguments.replicates,
                              max_tokens=arguments.max_tokens,
                              temperature=arguments.temperature,
                              limit=arguments.limit, fresh=arguments.fresh)
    if path is None:
        raise SystemExit('Nothing outstanding')
    print(f'{count:,} requests for {arguments.model} on '
          f'{provider_of(arguments.model)}')
    print(f'Written to {path}')
    print(f'\nUpload it in the provider console, endpoint {arguments.endpoint}, '
          f'then when the job finishes:')
    print(f'    python scripts/run.py ingest --model {arguments.model} '
          f'--file <downloaded results>.jsonl')
    return 0


# Define function to read a finished batch from the command line
def run_ingest(arguments):
    section('Batch ingest')
    read, failed, truncated, repeated, blocked = read_batch(
        model=arguments.model, source=arguments.file,
        temperature=arguments.temperature)
    print(f'{read:,} replies read into '
          f'{result_path(arguments.model, ADAPTATION_DIR).name}, {failed} failed')
    if repeated:
        print(f'{repeated:,} already collected and skipped')
    if blocked:
        print(f'{blocked} prompts were refused by the provider before reaching '
              f'the model, so they carry no reply')
    if truncated:
        print(f'{truncated} replies hit the {GENERATION["max_tokens"]} token cap, '
              f'so their length is censored rather than measured')
    cost = spent(arguments.model)
    if cost is not None:
        print(f'{USAGE["input"]:,} input and {USAGE["output"]:,} output tokens, '
              f'${cost:,.2f} at the standard rate, ${cost / 2:,.2f} batched')
    return 0


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
        print('Supplied, not generated')
        reply = arguments.reply
    else:
        print(f'Generating with {arguments.model} on {arguments.backend}')
        reply = ask(arguments.backend, arguments.model, row['prompt'])
    show('Returned', reply)

    section('Judgement')
    if arguments.verdict:
        print('Supplied, classifier not called')
        output = arguments.verdict
    else:
        print(f'Scoring with {arguments.judge}')
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
        print(f'  Unreadable, {problem}')
    for field, value in (verdict or {}).items():
        print(f'  {field:<20} {value}')

    section('Comparison')
    expected = row['expected_answer']
    observed = (verdict or {}).get('answer', '')
    deviation = compare(observed=observed, expected=expected)
    print(f'  Expected             {expected or "None, this is the control"}')
    print(f'  Observed             {observed or "not read"}')
    print(f'  Deviation            '
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
    parser.add_argument('stage',
                        choices=['check', 'generate', 'export', 'ingest'])
    parser.add_argument('--model', default='')
    parser.add_argument('--backend', default='ollama', choices=list(BACKENDS))
    parser.add_argument('--judge', default='')
    parser.add_argument('--replicates', type=int, default=GENERATION['replicates'])
    parser.add_argument('--max-tokens', type=int, default=GENERATION['max_tokens'])
    parser.add_argument('--temperature', type=float,
                        default=GENERATION['temperature'])
    parser.add_argument('--limit', type=int, default=0,
                        help='stop after this many replies, to time a pass')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='conversations handed to vLLM at once')
    parser.add_argument('--workers', type=int, default=1,
                        help='api calls in flight at once, for a provider with '
                             'no batch queue')
    parser.add_argument('--endpoint', default='/v1/responses',
                        help='the endpoint a batch job runs against')
    parser.add_argument('--file', default='',
                        help='the results file downloaded from the console')
    parser.add_argument('--force', action='store_true',
                        help='export the whole pass again, ignoring what is '
                             'already collected')
    parser.add_argument('--fresh', action='store_true',
                        help='export every prompt again, ignoring what is collected')
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
    if arguments.stage in ('generate', 'export', 'ingest') and not arguments.model:
        raise SystemExit(f'--model is needed to {arguments.stage}')
    if arguments.stage == 'generate':
        failures = run_generation(arguments)
    elif arguments.stage == 'export':
        failures = run_export(arguments)
    elif arguments.stage == 'ingest':
        failures = run_ingest(arguments)
    else:
        failures = run_check(arguments)

    if failures:
        print(f'\n{failures} failed this pass, run again to retry them')
