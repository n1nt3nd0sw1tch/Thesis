"""Generates one reply from a model, through whichever runtime is available.

Four are supported, and which one to use is a property of the machine rather
than of the experiment. vLLM batches on a GPU and is what a full pass uses.
Ollama serves a model locally and handles the quantisation the safeguard
classifier ships in, so it is the way to run the real judge on a laptop. MLX is
the fast option on Apple silicon. Transformers runs anywhere and is slowest.

A model is loaded once and held, since loading dominates the cost of a short
reply and a run puts thousands of prompts to the same model.
"""

import json
import os
import urllib.request

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

from settings import GENERATION

# Ollama serves an OpenAI-compatible endpoint on this machine
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

# How long to wait for one reply, in seconds. A reasoning model thinks before it
# answers, so this is generous.
TIMEOUT = 600

# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------

# Define function to generate one reply through a local Ollama server
def generate_ollama(model_id, messages, max_tokens, temperature):
    payload = {'model': model_id, 'messages': messages, 'stream': False,
               'options': {'temperature': temperature,
                           'top_p': GENERATION['top_p'],
                           'num_predict': max_tokens}}
    request = urllib.request.Request(
        f'{OLLAMA_URL}/api/chat', method='POST',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = json.loads(response.read())
    except urllib.error.URLError as problem:
        raise SystemExit(f'no Ollama server at {OLLAMA_URL}: {problem.reason}. '
                         f'Start it with `ollama serve` and fetch the model '
                         f'with `ollama pull {model_id}`.')
    # a reasoning model returns its thinking separately from its answer
    return str(body.get('message', {}).get('content', '')).strip()


# Define function to generate one reply with vLLM, which batches on a GPU
def generate_vllm(model_id, messages, max_tokens, temperature, loaded={}):
    from vllm import LLM, SamplingParams
    if model_id not in loaded:
        loaded[model_id] = LLM(model=model_id, trust_remote_code=True,
                               dtype='bfloat16')
    sampling = SamplingParams(temperature=temperature,
                              top_p=GENERATION['top_p'], max_tokens=max_tokens)
    output = loaded[model_id].chat(messages, sampling, use_tqdm=False)
    return output[0].outputs[0].text.strip()


# Define function to generate one reply with MLX on Apple silicon
def generate_mlx(model_id, messages, max_tokens, temperature, loaded={}):
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    if model_id not in loaded:
        loaded[model_id] = load(model_id)
    model, tokeniser = loaded[model_id]
    prompt = tokeniser.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    return generate(model, tokeniser, prompt=prompt, max_tokens=max_tokens,
                    sampler=make_sampler(temp=temperature,
                                         top_p=GENERATION['top_p']),
                    verbose=False).strip()


# Define function to generate one reply with transformers
def generate_transformers(model_id, messages, max_tokens, temperature, loaded={}):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_id not in loaded:
        tokeniser = AutoTokenizer.from_pretrained(model_id)
        # a gpu takes bfloat16 and the accelerate device map; a cpu takes
        # neither, so both paths are kept rather than requiring accelerate
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                         dtype=torch.float32)
        loaded[model_id] = (tokeniser, model)
    tokeniser, model = loaded[model_id]
    prompt = tokeniser.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokeniser(prompt, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_tokens,
                            do_sample=temperature > 0,
                            temperature=temperature or None,
                            top_p=GENERATION['top_p'],
                            pad_token_id=tokeniser.eos_token_id)
    return tokeniser.decode(output[0][inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True).strip()


BACKENDS = {'ollama': generate_ollama, 'vllm': generate_vllm,
            'mlx': generate_mlx, 'transformers': generate_transformers}


# Define function to generate one reply through the chosen backend
def generate(backend, model_id, messages, max_tokens=None, temperature=None):
    if backend not in BACKENDS:
        raise ValueError(f'{backend} is not one of {", ".join(BACKENDS)}')
    return BACKENDS[backend](
        model_id, messages,
        GENERATION['max_tokens'] if max_tokens is None else max_tokens,
        GENERATION['temperature'] if temperature is None else temperature)


# Define function to put one request to a model as a single turn
def ask(backend, model_id, prompt, max_tokens=None, temperature=None):
    return generate(backend, model_id, [{'role': 'user', 'content': prompt}],
                    max_tokens, temperature)
