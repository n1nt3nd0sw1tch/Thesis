"""Times two classifiers on the same fifty replies, and compares their verdicts.

    python test_judges.py
    python test_judges.py --model gpt-5.6-luna --n 50 --effort low
    python test_judges.py --effort high --tokens 4096

Run it from the repository root. Reads collected replies, writes nothing.

Two classifiers, one policy, one set of items.

    gpt-oss:120b-cloud            through the local Ollama daemon relaying to
                                  ollama.com. Six times larger, and a general
                                  model reading a policy rather than one trained
                                  to follow one, because Ollama's cloud serves no
                                  safeguard variant.

    openai/gpt-oss-safeguard-20b  through Groq. Purpose-trained for exactly this
                                  task: bring your own policy, and it reasons
                                  through it before answering.

Whether size or training wins here is an empirical question and nobody has
answered it for this rubric, so it is worth fifty calls before committing forty
six thousand.

The reasoning effort matters as much as the model. gpt-oss reasons in a hidden
channel before it answers, and that reasoning shares the output budget, so a cap
set too low cuts the model off mid-thought and no JSON arrives at all. The guide
for these models says not to cap output and to lower the effort instead, which is
what --effort does.

Needs GROQ_API_KEY in .env, and the Ollama daemon running and signed in.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, 'scripts')
import backends
import evaluate
import settings
import utils

GROQ = 'https://api.groq.com/openai/v1/chat/completions'

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to put one item to a classifier and time it. The policy goes in
# the system message and the item in the user message, which is what the harmony
# format expects and what the scoring stage already sends.
def ask(where, model, policy, item, effort, tokens):
    started = time.time()
    if where == 'ollama':
        payload = {'model': model, 'stream': False, 'keep_alive': '30m',
                   'messages': [{'role': 'system', 'content': policy},
                                {'role': 'user', 'content': item}],
                   'think': effort,
                   'options': {'temperature': 0.0, 'num_predict': tokens}}
        url, headers = f'{backends.OLLAMA_URL}/api/chat', {}
        key = utils.api_key('ollama')
    else:
        payload = {'model': model, 'temperature': 0.0,
                   'max_completion_tokens': tokens,
                   'reasoning_effort': effort,
                   'messages': [{'role': 'system', 'content': policy},
                                {'role': 'user', 'content': item}]}
        url, headers = GROQ, {}
        key = utils.api_key('groq')
    headers['Content-Type'] = 'application/json'
    if key:
        headers['Authorization'] = f'Bearer {key}'

    request = urllib.request.Request(url, method='POST',
                                     data=json.dumps(payload).encode(),
                                     headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as problem:
        return None, f'{problem.code}: {problem.read().decode()[:110]}', 0, 0
    except urllib.error.URLError as problem:
        return None, str(problem.reason)[:110], 0, 0

    took = time.time() - started
    if where == 'ollama':
        text = str((body.get('message') or {}).get('content') or '')
        tokens_used = body.get('prompt_eval_count', 0) + body.get('eval_count', 0)
    else:
        choices = body.get('choices') or [{}]
        text = str((choices[0].get('message') or {}).get('content') or '')
        tokens_used = (body.get('usage') or {}).get('total_tokens', 0)
    verdict, problems = evaluate.read(text)
    return verdict, '; '.join(problems), took, tokens_used


# Define function to draw a sample spread across the strata rather than off the
# top of the file, so refusals and compliances are both represented
def sample_replies(model, how_many):
    collected = utils.read_lines(utils.result_path(model, settings.ADAPTATION_DIR))
    if collected.empty:
        raise SystemExit(f'No replies collected for {model}')
    prompts = utils.read_table(settings.PROMPTS_PATH)
    benchmark = utils.read_table(settings.BENCHMARK_PATH)
    frame = (collected.merge(prompts[['prompt_id', 'scenario_id',
                                      'expected_answer']], on='prompt_id')
             .merge(benchmark[['scenario_id', 'scenario_type', 'request']],
                    on='scenario_id'))
    usable = frame[frame['response'].astype(str).str.strip() != '']
    per_stratum = -(-how_many // usable['scenario_type'].nunique())
    return (usable.sample(frac=1, random_state=settings.SEED)
            .groupby('scenario_type', group_keys=False)
            .head(per_stratum).head(how_many))


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='claude-haiku-4-5-20251001',
                        help='whose replies to judge')
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--effort', default='low',
                        choices=['low', 'medium', 'high'])
    parser.add_argument('--tokens', type=int, default=4096,
                        help='output budget, shared with the hidden reasoning')
    arguments = parser.parse_args()

    judges = [('ollama', 'gpt-oss:120b-cloud'),
              ('groq', 'openai/gpt-oss-safeguard-20b')]
    if not utils.api_key('groq'):
        print('No GROQ_API_KEY in .env, so only the Ollama route will run.\n')
        judges = judges[:1]

    sample = sample_replies(arguments.model, arguments.n)
    policy = evaluate.build_policy()
    print(f'Judging   {len(sample)} replies from {arguments.model}')
    print(f'Spread    {sample["scenario_type"].value_counts().to_dict()}')
    print(f'Policy    {len(policy) // 4:,} tokens')
    print(f'Effort    {arguments.effort}, output budget {arguments.tokens:,}\n')

    results = {}
    for where, model in judges:
        rows, failed, said = [], 0, ''
        started = time.time()
        for row in sample.itertuples():
            item = evaluate.build_item(row.request, row.response)
            verdict, problem, took, tokens = ask(where, model, policy, item,
                                                 arguments.effort,
                                                 arguments.tokens)
            if verdict is None or problem:
                failed += 1
                # keep the first reason, so a run that fails throughout says why
                said = said or problem
            rows.append({'prompt_id': row.prompt_id,
                         'expected': row.expected_answer,
                         'answer': (verdict or {}).get('answer', ''),
                         'seconds': took, 'tokens': tokens,
                         'problem': problem})
        wall = time.time() - started
        results[model] = rows
        times = [r['seconds'] for r in rows if r['seconds']]
        print(f'{model}')
        print(f'  {len(rows)} calls in {wall:,.0f}s, '
              f'{statistics.median(times) if times else 0:.2f}s each at the median')
        print(f'  {failed} could not be read, '
              f'{sum(r["tokens"] for r in rows) // max(len(rows), 1):,} tokens a call')
        if said:
            print(f'  first failure: {said}')
        matched = sum(r['answer'] == r['expected'] for r in rows if r['answer'])
        print(f'  {matched} of {len(rows)} matched the expected answer\n')

    # what a full pass would take at that rate
    per_model = len(utils.read_table(settings.PROMPTS_PATH)) * \
        settings.GENERATION['replicates']
    panel = per_model * sum(1 for m in settings.MODELS.values()
                            if m['access'] in ('api', 'local'))
    print(f'A pass is {per_model:,} replies a model, {panel:,} across the panel.')
    print(f"{'judge':<32} {'1 worker':>10} {'4':>8} {'8':>8} {'16':>8}")
    for model, rows in results.items():
        times = [r['seconds'] for r in rows if r['seconds']]
        rate = statistics.median(times) if times else 0
        hours = [f'{panel * rate / workers / 3600:,.1f}h' for workers in (1, 4, 8, 16)]
        print(f'{model:<32} ' + ' '.join(f'{h:>8}' for h in hours))

    if len(results) == 2:
        left, right = list(results)
        answers = {r['prompt_id']: r['answer'] for r in results[left]}
        agreed = [r for r in results[right]
                  if r['answer'] and answers.get(r['prompt_id'])
                  and r['answer'] == answers[r['prompt_id']]]
        both = [r for r in results[right]
                if r['answer'] and answers.get(r['prompt_id'])]
        print(f'\nThey agree on {len(agreed)} of {len(both)} replies both could read.')
        differ = [r for r in both if r['answer'] != answers[r['prompt_id']]]
        for row in differ[:8]:
            said = sample[sample['prompt_id'] == row['prompt_id']]
            print(f'\n  {row["prompt_id"]}  expected {row["expected"]}')
            print(f'    asked   {said["request"].iloc[0]}')
            print(f'    replied {str(said["response"].iloc[0])[:150]}...')
            print(f'    {left}: {answers[row["prompt_id"]]}   {right}: {row["answer"]}')

    print('\nRead the disagreements rather than counting them. Fifty is too few')
    print('for a rate, and the case this rubric turns on is a reply that declines')
    print('and then supplies the material, so look for that shape first.')