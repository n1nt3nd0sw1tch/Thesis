"""Collects replies to every prompt, one line at a time.

Each reply is appended to responses.jsonl as it arrives, and any prompt already
in that file is skipped, so a run that stops part way resumes where it left off
rather than starting again. That matters because a full pass is tens of thousands
of generations and will not finish in one sitting.

Four backends are supported. vLLM runs on a GPU node and is what a full pass
uses. Ollama serves a model on this machine and is the simplest to leave running
overnight. MLX runs on Apple silicon and is the fastest option on a Mac.
Transformers runs anywhere and is slowest.

    ollama pull qwen3:8b
    python scripts/run_generation.py --model qwen3:8b --backend ollama

    python scripts/run_generation.py --backend mlx \
        --model mlx-community/Qwen2.5-7B-Instruct-4bit --limit 60

Where run_batch.py samples whole scenarios to check the design reads, this
collects the lot. Progress prints once a minute rather than per reply, so the log
stays short enough to read the morning after.
"""

import argparse
import json
import os
import time
import urllib.request

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

import pandas as pd
from settings import (GENERATION, PROMPTS_PATH, ADAPTATION_DIR, RESPONSE_FIELDS,
                      append_line, read_all, read_lines, adaptation_path,
                      section, shape_of)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

# Stop after this many failures in a row. One awkward prompt is worth carrying
# on past; a backend that is not running is not.
GIVE_UP_AFTER = 20

# How often to report, in seconds. A line per reply would run to tens of
# thousands of lines and bury the failures worth seeing.
REPORT_EVERY = 60

# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------

# Define function to generate one reply through a local Ollama server
def generate_ollama(model_id, prompt, max_tokens, temperature):
    payload = {'model': model_id, 'stream': False,
               'messages': [{'role': 'user', 'content': prompt}],
               'options': {'temperature': temperature,
                           'top_p': GENERATION['top_p'],
                           'num_predict': max_tokens}}
    request = urllib.request.Request(
        f'{OLLAMA_URL}/api/chat', method='POST',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read())
    return str(body.get('message', {}).get('content', '')).strip()


# Define function to generate one reply with MLX on Apple silicon
def generate_mlx(model_id, prompt, max_tokens, temperature, cache={}):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    if model_id not in cache:
        cache[model_id] = load(model_id)
    model, tokeniser = cache[model_id]
    text = tokeniser.apply_chat_template([{'role': 'user', 'content': prompt}],
                                         tokenize=False, add_generation_prompt=True)
    return generate(model, tokeniser, prompt=text, max_tokens=max_tokens,
                    sampler=make_sampler(temp=temperature,
                                         top_p=GENERATION['top_p']),
                    verbose=False).strip()


# Define function to generate one reply with transformers
def generate_transformers(model_id, prompt, max_tokens, temperature, cache={}):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id not in cache:
        tokeniser = AutoTokenizer.from_pretrained(model_id)
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                         dtype=torch.float32)
        cache[model_id] = (tokeniser, model)
    tokeniser, model = cache[model_id]
    text = tokeniser.apply_chat_template([{'role': 'user', 'content': prompt}],
                                         tokenize=False, add_generation_prompt=True)
    inputs = tokeniser(text, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_tokens,
                            do_sample=temperature > 0,
                            temperature=temperature or None,
                            top_p=GENERATION['top_p'],
                            pad_token_id=tokeniser.eos_token_id)
    return tokeniser.decode(output[0][inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True).strip()


# Define function to generate replies with vLLM, which batches them on a GPU
def generate_vllm(model_id, prompt, max_tokens, temperature, cache={}):
    from vllm import LLM, SamplingParams
    if model_id not in cache:
        cache[model_id] = LLM(model=model_id, trust_remote_code=True,
                              dtype='bfloat16')
    sampling = SamplingParams(temperature=temperature,
                              top_p=GENERATION['top_p'], max_tokens=max_tokens)
    output = cache[model_id].chat([{'role': 'user', 'content': prompt}],
                                  sampling, use_tqdm=False)
    return output[0].outputs[0].text.strip()


BACKENDS = {'ollama': generate_ollama, 'mlx': generate_mlx,
            'transformers': generate_transformers, 'vllm': generate_vllm}

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to list the prompt and replicate pairs still to collect
def outstanding(prompts, collected, replicates):
    wanted = [(row.prompt_id, replicate) for row in prompts.itertuples()
              for replicate in range(1, replicates + 1)]
    if collected.empty:
        return wanted
    done = {(row.prompt_id, int(row.replicate)) for row in collected.itertuples()
            if not str(row.error).strip()}
    return [pair for pair in wanted if pair not in done]


# Define function to collect the outstanding replies, appending as they arrive
def collect(prompts, pending, backend, model, max_tokens, temperature, path,
            give_up_after=GIVE_UP_AFTER):
    by_id = prompts.set_index('prompt_id')['prompt']
    generate = BACKENDS[backend]
    started, spoke = time.time(), time.time()
    failures = consecutive = 0
    for index, (prompt_id, replicate) in enumerate(pending, start=1):
        try:
            reply = generate(model, by_id[prompt_id], max_tokens, temperature)
            error, consecutive = '', 0
        except Exception as problem:
            # one failure should not end an overnight run, so it is recorded
            # and the pass continues; a rerun retries whatever failed
            reply = ''
            error = f'{type(problem).__name__}: {problem}'
            failures, consecutive = failures + 1, consecutive + 1
        # the long field goes last so that the short ones line up when the
        # file is read directly rather than through pandas
        append_line(path, {'prompt_id': prompt_id, 'model': model,
                           'replicate': replicate, 'backend': backend,
                           'temperature': temperature, 'error': error,
                           'response': reply})
        if consecutive >= give_up_after:
            # a run this long means the backend is unreachable rather than one
            # prompt being awkward, and continuing would only fill the file
            # with errors
            print(f'  stopping after {consecutive} failures in a row')
            print(f'  last error: {error}')
            break
        if time.time() - spoke >= REPORT_EVERY or index == len(pending):
            spoke = time.time()
            rate = index / max(time.time() - started, 1)
            left = (len(pending) - index) / rate if rate else 0
            print(f'  {index} of {len(pending)}, {rate * 3600:.0f} an hour, '
                  f'{left / 3600:.1f} hours left, {failures} failed')
    return failures


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', default='ollama', choices=list(BACKENDS))
    parser.add_argument('--replicates', type=int, default=GENERATION['replicates'])
    parser.add_argument('--max-tokens', type=int, default=GENERATION['max_tokens'])
    parser.add_argument('--temperature', type=float,
                        default=GENERATION['temperature'])
    parser.add_argument('--limit', type=int, default=0,
                        help='stop after this many replies, for a timing check')
    args = parser.parse_args()

    section('Generation')
    prompts = pd.read_csv(PROMPTS_PATH, dtype=str, keep_default_na=False)
    path = adaptation_path(args.model)
    collected = read_lines(path)
    pending = outstanding(prompts=prompts, collected=collected,
                          replicates=args.replicates)
    wanted = len(prompts) * args.replicates
    already = wanted - len(pending)
    if args.limit:
        pending = pending[:args.limit]

    print(f'{args.model} on {args.backend}, {len(prompts)} prompts x '
          f'{args.replicates} replicates')
    print(f'writing to {path}')
    print(f'{already} of {wanted} already collected, {len(pending)} to go'
          + (f' (limited from {wanted - already})' if args.limit else ''))
    if not pending:
        raise SystemExit('nothing outstanding')

    failures = collect(prompts=prompts, pending=pending, backend=args.backend,
                       model=args.model, max_tokens=args.max_tokens,
                       temperature=args.temperature, path=path)

    section('Collected')
    everything = read_all(ADAPTATION_DIR)
    print(f'{shape_of(everything)} across {ADAPTATION_DIR.name}')
    print(everything.groupby('model').size().to_string())
    if failures:
        print(f'{failures} failed this pass, run again to retry them')
