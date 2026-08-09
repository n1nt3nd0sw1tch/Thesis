"""Checks every model in the panel before a run. Api models are looked up
against their provider so a renamed or retired identifier fails here rather
than part way through generation; local models are checked against the Hugging
Face Hub so a missing repository is caught before the cluster job starts."""

import pandas as pd
import requests
from settings import (GENERATION, JUDGE, MODELS, PROVIDER_KEYS, TOTAL_SCENARIOS,
                      api_key, section, shape_of)

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
# Functions
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


# Define function to report how many responses the panel implies
def report_cost(models, generation, prompts):
    calls = prompts * len(models) * generation['replicates']
    section('Run size')
    print(f'Prompts: {prompts}')
    print(f'Models: {len(models)}, replicates: {generation["replicates"]}')
    print(f'Responses: {calls}')
    print(f'Judged: {calls}')

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    section('Model panel')
    panel = check_panel(models=MODELS, judge=JUDGE)
    print(f'Panel: {shape_of(panel)}')
    print(panel.to_string(index=False))

    failed = panel[panel['status'] != 'ok']
    if len(failed):
        print(f'\n{len(failed)} models are not reachable')
