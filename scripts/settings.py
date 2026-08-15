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

# The implicit arm marks minority rather than a band, because no English
# education stage, workplace or peer group sits wholly inside 13 to 17. Minor
# spans Child and Adolescent, so it carries an expected answer only where those
# two agree, which the scenario types are built to ensure.
MINOR = 'Minor'
MINOR_BANDS = [band for band in AGE_BANDS if band != AGE_BANDS[-1]]
BANDS = AGE_BANDS + [MINOR]


# Define function to place an age in its band
def band_for_age(age):
    if age == '' or age is None:
        return ''
    age = int(age)
    for band, oldest in AGE_BAND_LIMITS.items():
        if age <= oldest:
            return band
    raise ValueError(f'age {age} falls outside every band')


# Define function to name a column after a measure
def measure_column(name):
    return name.lower().replace(' ', '_')


# Define function to fill in the one field a condition does not state. Every
# other field is written out in config/settings.yml rather than inferred, so a
# new kind of condition needs no change here.
def expand_condition(condition):
    age = condition.get('age', '')
    opener = condition.get('opener')
    if opener is None:
        opener = EXPLICIT_OPENER.format(age=age) if age != '' else ''
    return {'name': condition['name'],
            'age': age,
            'band': condition.get('band', ''),
            'signal': condition['signal'],
            'cue': condition['cue'],
            'opener': opener}


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

# Where a scenario has no source record, the author is named in its place.
AUTHORED = 'Author'
DATASET_NAMES = sorted({spec['name'] for spec in SOURCES.values()} | {AUTHORED})

# ----------------------------------------------------------------------------
# What each file carries
# ----------------------------------------------------------------------------

# Every cue is now an opening sentence, so a scenario carries one request and
# no variants of it. The request is what a system is asked in every condition.
# The candidate pool: one row per usable source record. Regenerated on every run
# and not committed, since the benchmark carries everything downstream needs.
DRAFTS_COLUMNS = ['source_id', 'dataset', 'domain', 'scenario_type',
                  'source_prompt', 'request']

# The benchmark is self-contained, so that the released file needs nothing else
# read alongside it. Provenance travels with the scenario rather than being
# asserted in prose: source_id names the record a scenario was adapted from,
# dataset names where that record came from or Author where the scenario was
# written for this benchmark, and source_prompt is the wording of that record
# before it was rewritten, blank where there was none. The two texts sit next to
# each other at the end, so the adaptation can be read across a single row.
BENCHMARK_COLUMNS = ['scenario_id', 'source_id', 'dataset', 'domain',
                     'scenario_type', 'source_prompt', 'request']

# The request is stored beside the prompt so that byte identity across
# conditions can be checked by reading the file rather than by rebuilding it.
PROMPT_COLUMNS = ['prompt_id', 'scenario_id', 'condition', 'age', 'band',
                  'signal', 'cue', 'opener', 'request', 'prompt',
                  'expected_answer']

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
assert all(condition['band'] in BANDS + [''] for condition in CONDITIONS)
assert all(condition['cue'] in CUES for condition in CONDITIONS)
assert all(condition['signal'] in ('Explicit', 'Implicit', 'None')
           for condition in CONDITIONS)
assert len(set(CONDITION_NAMES)) == len(CONDITIONS)
assert set(BANDS) >= {condition['band'] for condition in CONDITIONS
                      if condition['band']}
assert all(set(answers) == set(AGE_BANDS) for answers in TYPE_ANSWERS.values()
           if answers)
assert all(answer in ANSWERS for answers in TYPE_ANSWERS.values()
           for answer in answers.values())
assert all([ANSWERS.index(answers[band]) for band in AGE_BANDS]
           == sorted(ANSWERS.index(answer) for answer in answers.values())
           for answers in TYPE_ANSWERS.values() if answers)
assert len({tuple(answers[band] for band in AGE_BANDS)
            for answers in TYPE_ANSWERS.values() if answers}) \
    == len([answers for answers in TYPE_ANSWERS.values() if answers])
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
