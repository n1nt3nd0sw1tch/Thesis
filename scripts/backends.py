"""Generates one reply from a model, through whichever runtime is available.

Which runtime to use is a property of the machine rather than of the experiment,
except for api, which is a property of the model. vLLM batches on a GPU and is
what a full pass over an open-weight model uses. Ollama serves a model locally
and handles the quantisation the safeguard classifier ships in, so it is the way
to run the real judge on a laptop. MLX is the fast option on Apple silicon.
Transformers runs anywhere and is slowest. api reaches the proprietary systems,
which cannot be run locally at all, and routes to the right provider by looking
the model up in the panel.

A local model is loaded once and held, since loading dominates the cost of a
short reply and a run puts thousands of prompts to the same model. An api call
is retried with backoff, since rate limiting is the expected failure at volume
and losing a reply to it would leave a gap that a rerun has to fill.
"""

import json
import os
import random
import time
from threading import Lock
import urllib.error
import urllib.request

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

from settings import GENERATION, JUDGES, MODELS
from utils import api_key

# Ollama serves an OpenAI-compatible endpoint on this machine
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

# How long to wait for one reply, in seconds. A reasoning model thinks before it
# answers, so this is generous.
TIMEOUT = 600

# How many times to retry an api call, and how long to wait first. Each attempt
# waits twice as long as the last, with jitter so that concurrent runs do not
# retry in step. Rate limiting is the expected failure at four thousand calls a
# model, and it is temporary, so it is worth waiting out rather than recording.
RETRIES = 5
BACKOFF = 2.0
RETRY_ON = {408, 409, 429, 500, 502, 503, 504}

# What the run has consumed so far, updated by every api call and read by the
# stage that is running. Kept here rather than returned, so that the backend
# signature stays the same for a local model, which costs nothing.
USAGE = {'calls': 0, 'input': 0, 'output': 0}

# When each provider may next be called, and the lock held while that is read
# and set. A provider that publishes a rate limit refuses the surplus rather
# than queueing it, and the retry that follows costs more than the concurrency
# gained, so requests are spaced at the source instead of being retried.
_NEXT_CALL = {}
_PACING = Lock()


# Define function to wait until a model may be called again, if its entry in the
# panel names a rate. Called before the request rather than after a refusal,
# because a refused request has already cost its round trip.
def pace(model_id):
    rate = panel_entry(model_id, 'rate', 0)
    if not rate:
        return
    with _PACING:
        now = time.monotonic()
        ready = max(now, _NEXT_CALL.get(model_id, 0))
        _NEXT_CALL[model_id] = ready + 1 / float(rate)
    if ready > now:
        time.sleep(ready - now)


# The last call's reasoning tokens and stop reason. A live run has no result
# file to read them back from, and both decide whether a pass is usable.
LAST_REASONING = 0
LAST_FINISH = ''

# Concurrent workers all report into the same totals, and += on a dict value is
# a read and a write rather than one step, so without this the counts drift.
_COUNTING = Lock()

# Where each provider reports what a call consumed
USAGE_FIELDS = {
    'openai': ('usage', 'input_tokens', 'output_tokens'),
    'anthropic': ('usage', 'input_tokens', 'output_tokens'),
    'google': ('usageMetadata', 'promptTokenCount', 'candidatesTokenCount'),
    'deepseek': ('usage', 'prompt_tokens', 'completion_tokens'),
    'mistral': ('usage', 'prompt_tokens', 'completion_tokens'),
    'qwen': ('usage', 'prompt_tokens', 'completion_tokens'),
}

# Where each provider takes a conversation, and how it names the pieces. OpenAI
# is reached through the Responses API rather than chat completions, because the
# current models expose their reasoning effort only there.
PROVIDERS = {
    'openai': {
        'url': 'https://api.openai.com/v1/responses',
        'headers': lambda key: {'Authorization': f'Bearer {key}'},
    },
    'anthropic': {
        'url': 'https://api.anthropic.com/v1/messages',
        'headers': lambda key: {'x-api-key': key,
                                'anthropic-version': '2023-06-01'},
    },
    'google': {
        'url': 'https://generativelanguage.googleapis.com/v1beta/models/'
               '{model}:generateContent',
        'headers': lambda key: {'x-goog-api-key': key},
    },
    # DeepSeek serves the OpenAI chat completions dialect at its own host, and
    # has no batch queue, so this model is generated live rather than submitted
    'deepseek': {
        'url': 'https://api.deepseek.com/chat/completions',
        'headers': lambda key: {'Authorization': f'Bearer {key}'},
    },
    # Mistral serves the same dialect, and does have a batch queue
    'mistral': {
        'url': 'https://api.mistral.ai/v1/chat/completions',
        'headers': lambda key: {'Authorization': f'Bearer {key}'},
    },
    # Qwen serves the same dialect from DashScope's compatible endpoint
    'qwen': {
        'url': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/'
               'chat/completions',
        'headers': lambda key: {'Authorization': f'Bearer {key}'},
    },
}

# The providers that speak the OpenAI chat completions dialect at their own host
CHAT_COMPLETIONS = {'deepseek', 'mistral', 'qwen'}

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
        raise SystemExit(f'No Ollama server at {OLLAMA_URL}: {problem.reason}. '
                         f'Start it with `ollama serve` and fetch the model '
                         f'with `ollama pull {model_id}`.')
    # a reasoning model returns its thinking separately from its answer
    return str(body.get('message', {}).get('content', '')).strip()


# Define function to generate one reply with vLLM, which batches on a GPU
def generate_vllm(model_id, messages, max_tokens, temperature, loaded={}):
    return generate_vllm_batch(model_id, [messages], max_tokens, temperature)[0]


# Define function to generate a reply to each of many conversations at once.
# vLLM schedules them together and keeps the GPU fed, which is the whole reason
# to ask for one: putting a single conversation at a time leaves it idle between
# tokens and turns a night's work into a week's.
def generate_vllm_batch(model_id, conversations, max_tokens, temperature,
                        loaded={}):
    from vllm import LLM, SamplingParams
    if model_id not in loaded:
        loaded[model_id] = LLM(model=model_id, trust_remote_code=True,
                               dtype='bfloat16')
    sampling = SamplingParams(temperature=temperature,
                              top_p=GENERATION['top_p'], max_tokens=max_tokens)
    outputs = loaded[model_id].chat(list(conversations), sampling, use_tqdm=False)
    return [output.outputs[0].text.strip() for output in outputs]


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


# ----------------------------------------------------------------------------
# The proprietary systems
# ----------------------------------------------------------------------------

# Define function to read one field of a model's entry in the panel
def panel_entry(model_id, field, default=None):
    for spec in list(MODELS.values()) + list(JUDGES.values()):
        if spec['id'] == model_id:
            return spec.get(field, default)
    return default


# Define function to say whether a model accepts the sampling parameters the
# design asks for. Some reject them, and a request carrying one is refused
# outright rather than ignored.
def takes_sampling(model_id):
    return str(panel_entry(model_id, 'sampling', '')).lower() != 'provider'


# Define function to read whether reasoning is to be switched off. The panel
# setting applies to every model; a model may override it, which is how one arm
# is run both ways for the comparison rather than by editing the panel.
def reasoning_off(model_id):
    setting = str(panel_entry(model_id, 'reasoning', '')
                  or GENERATION.get('reasoning', 'provider')).strip().lower()
    return setting == 'none'


# Define function to find which provider serves a model, from the panel rather
# than from the identifier, so that a renamed model needs no change here
def provider_of(model_id):
    for spec in list(MODELS.values()) + list(JUDGES.values()):
        if spec['id'] == model_id:
            return spec['provider']
    raise ValueError(f'{model_id} is not in the panel, so its provider is '
                     f'unknown. Add it to config/settings.yml under models.')


# Define function to shape one conversation the way a provider expects it. All
# three take the same messages and differ only in where a system turn goes and
# what the fields are called.
def build_payload(provider, model_id, messages, max_tokens, temperature):
    system = ' '.join(m['content'] for m in messages if m['role'] == 'system')
    turns = [m for m in messages if m['role'] != 'system']
    off = reasoning_off(model_id)
    if provider == 'openai':
        # Sampling is sent where the model accepts it, so that as much of the
        # panel as possible is decoded the same way. Where it is refused the
        # model runs at the provider's own defaults, which is recorded in the
        # panel rather than left to be inferred from a failed batch.
        payload = {'model': model_id, 'max_output_tokens': max_tokens,
                   'input': [{'role': m['role'], 'content': m['content']}
                             for m in turns]}
        if takes_sampling(model_id):
            payload['temperature'] = temperature
            payload['top_p'] = GENERATION['top_p']
        if system:
            payload['instructions'] = system
        # reasoning tokens are billed as output and count against the cap, so a
        # model left to think by default can spend the whole budget before it
        # writes anything
        if off:
            payload['reasoning'] = {'effort': 'none'}
        return payload
    if provider in CHAT_COMPLETIONS:
        payload = {'model': model_id, 'max_tokens': max_tokens,
                   'messages': [{'role': m['role'], 'content': m['content']}
                                for m in messages]}
        # Each names the switch differently: DeepSeek takes a thinking block,
        # Qwen a flag, Mistral an effort level
        if off:
            if provider == 'deepseek':
                payload['thinking'] = {'type': 'disabled'}
            elif provider == 'qwen':
                payload['enable_thinking'] = False
            else:
                payload['reasoning_effort'] = 'none'
        # DeepSeek ignores these in thinking mode and honours them out of it, so
        # they are sent whenever the model has not declared that it refuses them
        if takes_sampling(model_id):
            payload['temperature'] = temperature
            payload['top_p'] = GENERATION['top_p']
        return payload
    if provider == 'anthropic':
        # Anthropic asks that only one of temperature and top_p be set, so the
        # one the design names is sent and the other left at its default
        payload = {'model': model_id, 'max_tokens': max_tokens,
                   'temperature': temperature,
                   'messages': [{'role': m['role'], 'content': m['content']}
                                for m in turns]}
        if system:
            payload['system'] = system
        # Extended thinking is opt in here, so off is the state of not asking
        return payload
    payload = {'contents': [{'role': 'user' if m['role'] == 'user' else 'model',
                             'parts': [{'text': m['content']}]} for m in turns],
               'generationConfig': {'maxOutputTokens': max_tokens,
                                    'temperature': temperature,
                                    'topP': GENERATION['top_p']}}
    if off:
        payload['generationConfig']['thinkingConfig'] = {'thinkingBudget': 0}
    if system:
        payload['systemInstruction'] = {'parts': [{'text': system}]}
    return payload


# Define function to read the reply out of whatever the provider returned
def read_reply(provider, body):
    if provider in CHAT_COMPLETIONS:
        choices = body.get('choices') or []
        if not choices:
            return ''
        content = choices[0].get('message', {}).get('content') or ''
        # With reasoning on, the content arrives as a list of blocks rather than
        # a string, and the thinking is one of them. Taking the whole list would
        # put the model's deliberation into the text that gets scored.
        if isinstance(content, list):
            content = ''.join(part.get('text', '') if isinstance(part, dict)
                              and part.get('type') != 'thinking' else ''
                              for part in content)
        return str(content).strip()
    if provider == 'openai':
        if body.get('output_text'):
            return str(body['output_text']).strip()
        # a reasoning model returns its thinking as a separate block, so only
        # the message blocks are read
        parts = [content.get('text', '')
                 for item in body.get('output', []) if item.get('type') == 'message'
                 for content in item.get('content', [])
                 if content.get('type') == 'output_text']
        return ''.join(parts).strip()
    if provider == 'anthropic':
        return ''.join(block.get('text', '')
                       for block in body.get('content', [])
                       if block.get('type') == 'text').strip()
    candidates = body.get('candidates', [])
    if not candidates:
        return ''
    # a thought arrives as a part flagged thought, beside the answer parts, so
    # taking every part would fold the thinking into the text that gets scored
    return ''.join(part.get('text', '') for part
                   in candidates[0].get('content', {}).get('parts', [])
                   if not part.get('thought')).strip()


# Define function to add what one call consumed to the running total
def record_usage(provider, body):
    global LAST_REASONING, LAST_FINISH
    field, sent, received = USAGE_FIELDS[provider]
    usage = body.get(field, {}) or {}
    LAST_REASONING = int((usage.get('completion_tokens_details') or {})
                         .get('reasoning_tokens', 0)
                         or (usage.get('output_tokens_details') or {})
                         .get('reasoning_tokens', 0)
                         or usage.get('thoughtsTokenCount', 0) or 0)
    LAST_FINISH = str(next((c.get('finish_reason') or c.get('finishReason') or ''
                            for c in (body.get('choices')
                                      or body.get('candidates') or [])), ''))
    with _COUNTING:
        USAGE['calls'] += 1
        USAGE['input'] += int(usage.get(sent, 0) or 0)
        # Google reports thinking separately but bills it as output, so it is
        # added here rather than leaving the cost understated
        USAGE['output'] += int(usage.get(received, 0) or 0) \
            + int(usage.get('thoughtsTokenCount', 0) or 0)


# Define function to price what has been consumed so far, in pounds of nothing
# if the model is local and carries no rate in the panel
def spent(model_id, usage=None):
    price = panel_entry(model_id, 'price')
    usage = USAGE if usage is None else usage
    if not price:
        return None
    return (usage['input'] * price['input']
            + usage['output'] * price['output']) / 1e6


# Define function to generate one reply through a provider's api, returning the
# whole body rather than the text. A live run records what a batch job would
# have returned, so that the two routes leave the same evidence behind.
def call_api(model_id, messages, max_tokens, temperature):
    pace(model_id)
    provider = provider_of(model_id)
    spec = PROVIDERS[provider]
    key = api_key(provider)
    if not key:
        raise SystemExit(f'No api key for {provider}. Put it in .env as '
                         f'{provider.upper()}_API_KEY, which is not committed.')

    payload = build_payload(provider, model_id, messages, max_tokens, temperature)
    request = urllib.request.Request(
        spec['url'].format(model=model_id), method='POST',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', **spec['headers'](key)})

    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body = json.loads(response.read())
            record_usage(provider, body)
            return body
        except urllib.error.HTTPError as problem:
            detail = problem.read().decode('utf-8', 'replace')[:200]
            if problem.code not in RETRY_ON or attempt == RETRIES - 1:
                raise RuntimeError(f'{provider} returned {problem.code}: {detail}')
        except urllib.error.URLError as problem:
            if attempt == RETRIES - 1:
                raise RuntimeError(f'{provider} unreachable: {problem.reason}')
        time.sleep(BACKOFF * (2 ** attempt) * (0.5 + random.random()))


# Define function to generate one reply through a provider's api
def generate_api(model_id, messages, max_tokens, temperature):
    return read_reply(provider_of(model_id),
                      call_api(model_id, messages, max_tokens, temperature))


BACKENDS = {'api': generate_api, 'ollama': generate_ollama,
            'vllm': generate_vllm, 'mlx': generate_mlx,
            'transformers': generate_transformers}


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


# Which backends can take many conversations at once, and how many to hand over
# in one go. Only vLLM schedules them together; the rest are one at a time.
BATCHED = {'vllm': generate_vllm_batch}
BATCH_SIZE = 64


# Define function to generate a reply to each of many conversations, through a
# backend that can take them together
def generate_many(backend, model_id, conversations, max_tokens=None,
                  temperature=None):
    if backend not in BATCHED:
        raise ValueError(f'{backend} takes one conversation at a time')
    return BATCHED[backend](
        model_id, conversations,
        GENERATION['max_tokens'] if max_tokens is None else max_tokens,
        GENERATION['temperature'] if temperature is None else temperature)
