"""The design, and where everything lives.

Reads config/settings.yml and config/scenarios.yml, and exposes what they state
along with everything that follows from it: the age conditions in full, the
expected answer of each scenario type, the columns each file carries. Nothing
here does any work. The machinery that acts on these values is in utils.py,
which imports this module rather than the other way round.
"""

from pathlib import Path

import yaml

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'config'
SETTINGS_PATH = CONFIG_DIR / 'settings.yml'
SCENARIOS_PATH = CONFIG_DIR / 'scenarios.yml'
ENV_PATH = ROOT / '.env'

# Data is kept in two directories that follow the order of the work. Sources
# holds what was downloaded and is never edited. Benchmark holds what the
# benchmark is made of, ending in the prompts put to a system. Both are fixed
# once the benchmark is frozen.
DATA_DIR = ROOT / 'data'

SOURCES_DIR = DATA_DIR / 'sources'
ORIGINAL_DIR = SOURCES_DIR / 'original'
DOWNLOADS_PATH = SOURCES_DIR / 'downloads.md'
AOA_PATH = ORIGINAL_DIR / 'aoa.csv'

BENCHMARK_DIR = DATA_DIR / 'benchmark'
DRAFTS_PATH = BENCHMARK_DIR / 'drafts.csv'
BENCHMARK_PATH = BENCHMARK_DIR / 'benchmark.csv'
PROMPTS_PATH = BENCHMARK_DIR / 'prompts.csv'
SCORES_PATH = BENCHMARK_DIR / 'scores.csv'

# Everything a run produces, kept apart from the data because it grows with
# every run while the benchmark stays fixed. One subfolder per experiment, named
# as Chapter 4 names them, and within each one file per model, so a run can be
# repeated or discarded without touching the others.
RESULTS_DIR = ROOT / 'results'
ADAPTATION_DIR = RESULTS_DIR / 'adaptation'
PERSISTENCE_DIR = RESULTS_DIR / 'persistence'
JUDGEMENTS_DIR = RESULTS_DIR / 'judgements'

JUDGEMENTS_PATH = RESULTS_DIR / 'judgements.csv'
DIALOGUES_PATH = PERSISTENCE_DIR / 'dialogues.csv'

DATA_DIRS = [SOURCES_DIR, ORIGINAL_DIR, BENCHMARK_DIR, RESULTS_DIR,
             ADAPTATION_DIR, PERSISTENCE_DIR, JUDGEMENTS_DIR]

# ----------------------------------------------------------------------------
# Read from config/
# ----------------------------------------------------------------------------

with open(SETTINGS_PATH, encoding='utf-8') as file:
    SETTINGS = yaml.safe_load(file)

# The scenarios are the one part of the design written by hand, so they are kept
# in a file of their own.
with open(SCENARIOS_PATH, encoding='utf-8') as file:
    SCENARIO_SETTINGS = yaml.safe_load(file)

SEED = SETTINGS['seed']
DOMAINS = SETTINGS['domains']
TYPES = SETTINGS['types']
CUES = SETTINGS['cues']
ANSWERS = SETTINGS['answers']
AGE_BAND_LIMITS = SETTINGS['age_bands']
EXPLICIT_OPENER = SETTINGS['explicit_opener']
SAFETY = SETTINGS['safety']
LANGUAGE = SETTINGS['language']
METHODS = SETTINGS['methods']
PERSISTENCE = SETTINGS['persistence']

DATASETS = SETTINGS['datasets']
SOURCES = SETTINGS['sources']
SAFECHILD_COMMIT = SETTINGS['safechild_commit']

GENERATION = SETTINGS['generation']
PROVIDER_KEYS = SETTINGS['keys']
MODELS = SETTINGS['models']
JUDGES = SETTINGS['judges']
JUDGE = JUDGES['primary']

SCENARIOS = SCENARIO_SETTINGS['scenarios']
WHO = SCENARIO_SETTINGS['who']

# ----------------------------------------------------------------------------
# The age conditions
# ----------------------------------------------------------------------------

AGE_BANDS = list(AGE_BAND_LIMITS)


# Define function to place an age in its band
def band_for_age(age):
    if age == '' or age is None:
        return ''
    age = int(age)
    for band, oldest in AGE_BAND_LIMITS.items():
        if age <= oldest:
            return band
    raise ValueError(f'age {age} falls outside every band')


# Define function to name the request variant a band uses. The canonical request
# and one variant per band. Each variant carries an age cue in place of the
# neutral phrase the canonical request holds. The wording runs source_prompt, as
# the corpus had it, to request, as rewritten here, to prompt, as put to a model.
def variant_column(band):
    return f'implicit_{band.lower()}'


# Define function to name a column after a measure
def measure_column(name):
    return name.lower().replace(' ', '_')


# Define function to fill in everything a condition implies from what it states.
# An explicit condition states an age and the rest follows from it; an implicit
# condition names a band and takes its request variant from that band and its
# cue family from the scenario; the control states nothing.
def expand_condition(condition):
    age = condition.get('age', '')
    band = condition.get('band') or band_for_age(age)
    signal = 'Explicit' if age else ('Implicit' if band else 'None')
    return {'name': condition['name'],
            'age': age,
            'band': band,
            'signal': signal,
            'cue': {'Explicit': 'Age', 'Implicit': '', 'None': 'None'}[signal],
            'variant': band if signal == 'Implicit' else '',
            'opener': EXPLICIT_OPENER.format(age=age) if age else ''}


CONDITIONS = [expand_condition(condition) for condition in SETTINGS['conditions']]
CONDITION_NAMES = [condition['name'] for condition in CONDITIONS]
CONDITION_AGES = [condition['age'] for condition in CONDITIONS]
SIGNALS = sorted({condition['signal'] for condition in CONDITIONS})

# ----------------------------------------------------------------------------
# Derived from the settings
# ----------------------------------------------------------------------------

DOMAIN_NAMES = {code: values['name'] for code, values in DOMAINS.items()}
DOMAIN_CODES = {name: code for code, name in DOMAIN_NAMES.items()}
TYPE_ANSWERS = {name: values['answers'] for name, values in TYPES.items()}
PER_DOMAIN = sum(values['count'] for values in TYPES.values())
TOTAL_SCENARIOS = PER_DOMAIN * len(DOMAINS)

DATASET_NAMES = sorted({spec['name'] for spec in SOURCES.values()})
KEEP_VALUES = ['yes', '']

# ----------------------------------------------------------------------------
# What each file carries
# ----------------------------------------------------------------------------

VARIANT_COLUMNS = (['request', 'implicit_cue']
                   + [variant_column(band) for band in AGE_BANDS])

DRAFTS_COLUMNS = ['source_id', 'dataset', 'domain', 'scenario_type',
                  'source_prompt', *VARIANT_COLUMNS, 'keep']
BENCHMARK_COLUMNS = ['scenario_id', 'dataset', 'domain', 'scenario_type',
                     *VARIANT_COLUMNS]
PROMPT_COLUMNS = ['prompt_id', 'scenario_id', 'condition', 'age', 'band',
                  'signal', 'cue', 'prompt', 'expected_answer']
SCORE_COLUMNS = ['variant', 'scenario_id', 'words', 'fkgl', 'fre', 'mean_aoa',
                 'max_aoa', 'difficult', 'covered']

# One record per reply. The identifying fields come first so that scanning the
# file shows what each line is without reading the reply itself, and the reply
# comes last because it is the only field of unpredictable length.
RESPONSE_COLUMNS = ['prompt_id', 'model', 'replicate', 'backend', 'temperature',
                    'error', 'response']

# One row per scored reply: what the model did with the request, the five safety
# measures, and the language measures computed from the text.
JUDGEMENT_COLUMNS = (['prompt_id', 'model', 'replicate', 'judge', 'answer']
                     + [measure_column(name) for name in SAFETY]
                     + [measure_column(name) for name in LANGUAGE]
                     + ['expected_answer', 'deviation'])

# One row per turn of a replayed dialogue. The first assistant turn is a reply
# already collected single turn, so only the later turns are generated. The
# method names how the user presses after that reply.
DIALOGUE_COLUMNS = ['dialogue_id', 'prompt_id', 'scenario_id', 'condition',
                    'band', 'model', 'opening_replicate', 'method', 'turn',
                    'role', 'text', 'expected_answer']

# ----------------------------------------------------------------------------
# Checks applied to the settings when this module is imported
# ----------------------------------------------------------------------------

assert all(condition['band'] == band_for_age(condition['age'])
           for condition in CONDITIONS if condition['age'] != '')
assert len(set(CONDITION_NAMES)) == len(CONDITIONS)
assert set(AGE_BANDS) >= {condition['band'] for condition in CONDITIONS
                          if condition['band']}
assert all(len(answers) == len(AGE_BANDS) for answers in TYPE_ANSWERS.values())
assert all(answer in ANSWERS for answers in TYPE_ANSWERS.values()
           for answer in answers)
assert all([ANSWERS.index(answer) for answer in answers]
           == sorted(ANSWERS.index(answer) for answer in answers)
           for answers in TYPE_ANSWERS.values())
assert len({tuple(answers) for answers in TYPE_ANSWERS.values()}) == len(TYPES)
assert all({'kind', 'name', 'licence'} <= set(spec) for spec in DATASETS.values())
assert all({'file', 'name', 'scenario_type', 'text', 'label', 'domains'}
           <= set(spec) for spec in SOURCES.values())
assert all(spec['scenario_type'] in TYPES for spec in SOURCES.values())
assert all(code in DOMAINS for spec in SOURCES.values()
           for code in spec['domains'])
assert all('filename' in spec if spec['kind'] == 'file' else 'origin' in spec
           for spec in DATASETS.values())
assert all(spec['kind'] in ('hub', 'url', 'file') for spec in DATASETS.values())
assert all({'provider', 'id', 'access', 'weights'} <= set(spec)
           for spec in [*MODELS.values(), *JUDGES.values()])
assert all(spec['access'] in ('api', 'local') for spec in MODELS.values())
assert all(spec['provider'] in PROVIDER_KEYS for spec in MODELS.values()
           if spec['access'] == 'api')
assert all(method in METHODS for method in PERSISTENCE['methods'])
assert all(len(values) >= 2 for values in SAFETY.values())
assert len({len(spec['turns']) for spec in METHODS.values()}) == 1
