"""Runs one prompt end to end: generates a reply with a local model, scores it
with the safeguard classifier, and compares the answer against the expectation.

The point is to check the plumbing on a single item before committing to a full
run. Three backends are supported. Transformers runs anywhere and is slow, vLLM
runs on a GPU node and is what the real generation uses, and echo generates
nothing and is there to test the scoring path on its own.

    python scripts/run_test.py --prompt-id sub-h1-age09
    python scripts/run_test.py --model Qwen/Qwen3.6-27B --backend vllm
    python scripts/run_test.py --reply "I can't help with that."

The safeguard classifier is MXFP4 quantised, so it needs accelerate installed and
a GPU that supports the format. Pass --judge to substitute a smaller classifier
while checking the plumbing, or --verdict to skip the classifier altogether and
test the scoring path on a supplied output.
"""

import argparse
import os
import textwrap

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import pandas as pd
from judge import build_item, build_policy, compare, describe, read
from settings import (BENCHMARK_PATH, GENERATION, JUDGE, PROMPTS_PATH, section)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

# A small model that runs on a CPU, so the plumbing can be checked without a GPU
DEFAULT_MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'

# The classifier reasons before answering, so it needs room beyond the reply
JUDGE_TOKENS = 1024

# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------

# Define function to generate one reply with transformers
def generate_transformers(model_id, messages, max_tokens, cache={}):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id not in cache:
        tokeniser = AutoTokenizer.from_pretrained(model_id)
        # a GPU takes bfloat16 and the accelerate device map; a CPU takes
        # neither, so both paths are kept rather than requiring accelerate
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                         dtype=torch.float32)
        cache[model_id] = (tokeniser, model)
    tokeniser, model = cache[model_id]
    prompt = tokeniser.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokeniser(prompt, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_tokens,
                            do_sample=GENERATION['temperature'] > 0,
                            temperature=GENERATION['temperature'],
                            top_p=GENERATION['top_p'],
                            pad_token_id=tokeniser.eos_token_id)
    return tokeniser.decode(output[0][inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True).strip()


# Define function to generate one reply with vLLM
def generate_vllm(model_id, messages, max_tokens, cache={}):
    from vllm import LLM, SamplingParams
    if model_id not in cache:
        cache[model_id] = LLM(model=model_id, trust_remote_code=True)
    sampling = SamplingParams(temperature=GENERATION['temperature'],
                              top_p=GENERATION['top_p'],
                              max_tokens=max_tokens)
    return cache[model_id].chat(messages, sampling)[0].outputs[0].text.strip()


BACKENDS = {'transformers': generate_transformers, 'vllm': generate_vllm}


# Define function to generate one reply with the chosen backend
def generate(backend, model_id, messages, max_tokens):
    if backend not in BACKENDS:
        raise ValueError(f'{backend} is not one of {", ".join(BACKENDS)}')
    return BACKENDS[backend](model_id, messages, max_tokens)


# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to pick the prompt to run, with its canonical request
def load_prompt(prompt_id, prompts_path, benchmark_path):
    prompts = pd.read_csv(prompts_path, dtype=str, keep_default_na=False)
    benchmark = pd.read_csv(benchmark_path, dtype=str, keep_default_na=False)
    chosen = prompts[prompts['prompt_id'] == prompt_id] if prompt_id else prompts
    if chosen.empty:
        raise KeyError(f'{prompt_id} is not in {prompts_path.name}')
    row = chosen.iloc[0]
    scenario = benchmark.set_index('scenario_id').loc[row['scenario_id']]
    return row, scenario


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
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--judge', default=JUDGE['id'])
    parser.add_argument('--backend', default='transformers',
                        choices=list(BACKENDS) + ['echo'])
    parser.add_argument('--reply', default='',
                        help='skip generation and score this text instead')
    parser.add_argument('--verdict', default='',
                        help='skip the classifier and read this output instead')
    parser.add_argument('--max-tokens', type=int, default=GENERATION['max_tokens'])
    args = parser.parse_args()

    row, scenario = load_prompt(prompt_id=args.prompt_id,
                                prompts_path=PROMPTS_PATH,
                                benchmark_path=BENCHMARK_PATH)

    section('Prompt')
    print(f'{row["prompt_id"]}  {scenario["domain"]} / '
          f'{scenario["scenario_type"]} / {row["condition"]}')
    show('Sent to the model', row['prompt'])

    section('Reply')
    if args.reply or args.backend == 'echo':
        reply = args.reply or '(nothing generated)'
        print(f'supplied, not generated')
    else:
        print(f'generating with {args.model} via {args.backend}')
        reply = generate(backend=args.backend, model_id=args.model,
                         max_tokens=args.max_tokens,
                         messages=[{'role': 'user', 'content': row['prompt']}])
    show('Returned', reply)

    section('Judgement')
    if args.verdict:
        print('output supplied, classifier not called')
        output = args.verdict
    elif args.backend == 'echo':
        print('echo backend, classifier not called')
        output = ''
    else:
        print(f'scoring with {args.judge}')
        # the classifier sees the canonical request, never the cued variant or
        # the opener, so it cannot infer which condition produced the reply
        output = generate(backend=args.backend, model_id=args.judge,
                          max_tokens=JUDGE_TOKENS,
                          messages=[{'role': 'system', 'content': build_policy()},
                                    {'role': 'user', 'content': build_item(
                                        scenario['request'], reply)}])
    verdict, problems = read(output)

    if problems:
        for problem in problems:
            print(f'  unreadable, {problem}')
    if verdict:
        for field, value in verdict.items():
            print(f'  {field:<20} {value}')

    section('Comparison')
    expected = row['expected_answer']
    observed = (verdict or {}).get('answer', '')
    deviation = compare(observed=observed, expected=expected)
    print(f'  expected             {expected or "none, this is the control"}')
    print(f'  observed             {observed or "not read"}')
    print(f'  deviation            {deviation if deviation is not None else "n/a"}'
          f'  ({describe(deviation)})')
