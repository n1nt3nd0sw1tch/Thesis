"""Puts one prompt to a model, scores the reply, and compares it against the
expectation.

The point is to check the plumbing on a single item before committing to a run
of tens of thousands. Every stage can be short-circuited: a supplied reply skips
generation, and a supplied verdict skips the classifier, so the scoring path can
be checked without loading a twenty billion parameter model.

    ollama pull gpt-oss-safeguard:20b
    python scripts/run_test.py --prompt-id sub-h1-age09 --backend ollama \
        --model qwen3:8b

    python scripts/run_test.py --reply "I cannot help with that."
"""

import argparse
import textwrap

import pandas as pd
from backends import BACKENDS, ask, generate
from judge import build_item, build_policy, compare, describe, read
from settings import BENCHMARK_PATH, JUDGE, PROMPTS_PATH, section

# The classifier reasons before it answers, so it needs room beyond the reply
JUDGE_TOKENS = 1024

# Scoring is a measurement and should not vary between runs, so the classifier
# is decoded greedily whatever temperature the evaluated systems use
JUDGE_TEMPERATURE = 0.0

# The identifier Ollama serves the classifier under, which differs from the hub
OLLAMA_JUDGE = 'gpt-oss-safeguard:20b'

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to load one prompt with the canonical request behind it
def load_prompt(prompt_id, prompts_path, benchmark_path):
    prompts = pd.read_csv(prompts_path, dtype=str, keep_default_na=False)
    benchmark = pd.read_csv(benchmark_path, dtype=str, keep_default_na=False)
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


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-id', default='')
    parser.add_argument('--backend', default='ollama', choices=list(BACKENDS))
    parser.add_argument('--model', default='')
    parser.add_argument('--judge', default='')
    parser.add_argument('--reply', default='',
                        help='skip generation and score this text instead')
    parser.add_argument('--verdict', default='',
                        help='skip the classifier and read this output instead')
    arguments = parser.parse_args()
    if not arguments.judge:
        arguments.judge = (OLLAMA_JUDGE if arguments.backend == 'ollama'
                           else JUDGE['id'])

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
        if not arguments.model:
            raise SystemExit('--model is needed unless --reply is given')
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
