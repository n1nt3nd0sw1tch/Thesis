"""Loads the benchmark design, the dataset settings and the model panel, then
provides the helpers the scripts share: identifier construction, api keys, and
validation."""

import json
import os
import re
from pathlib import Path

import pandas as pd
import yaml

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'config'
SCENARIOS_PATH = CONFIG_DIR / 'scenarios.yml'
ENV_PATH = ROOT / '.env'
# Data is kept in four directories that follow the order of the work. Sources
# holds what was downloaded and is never edited. Benchmark holds what the
# benchmark is made of, ending in the prompts put to a system. Responses holds
# what came back, and results holds what was made of it. Anything under sources
# and benchmark is fixed once the benchmark is frozen; anything under responses
# and results grows as the runs proceed.
DATA_DIR = ROOT / 'data'

SOURCES_DIR = DATA_DIR / 'sources'
ORIGINAL_DIR = SOURCES_DIR / 'original'
DOWNLOADS_PATH = SOURCES_DIR / 'downloads.md'

BENCHMARK_DIR = DATA_DIR / 'benchmark'
DRAFTS_PATH = BENCHMARK_DIR / 'drafts.csv'
BENCHMARK_PATH = BENCHMARK_DIR / 'benchmark.csv'
PROMPTS_PATH = BENCHMARK_DIR / 'prompts.csv'
SCORES_PATH = BENCHMARK_DIR / 'scores.csv'
LEXICAL_PATH = BENCHMARK_DIR / 'lexical.csv'

# Everything a run produces, kept apart from the data because it grows with
# every run while the benchmark stays fixed. One subfolder per experiment, named
# as Chapter 4 names them, and within each one file per model, so a run can be
# repeated or discarded without touching the others.
RESULTS_DIR = ROOT / 'results'
ADAPTATION_DIR = RESULTS_DIR / 'adaptation'
RECOGNITION_DIR = RESULTS_DIR / 'recognition'
PERSISTENCE_DIR = RESULTS_DIR / 'persistence'

JUDGEMENTS_PATH = RESULTS_DIR / 'judgements.csv'
JUDGED_PATH = RESULTS_DIR / 'judged.csv'

DATA_DIRS = [SOURCES_DIR, ORIGINAL_DIR, BENCHMARK_DIR, RESULTS_DIR,
             ADAPTATION_DIR, RECOGNITION_DIR, PERSISTENCE_DIR]

# ----------------------------------------------------------------------------
# Settings, read from config/
# ----------------------------------------------------------------------------

with open(CONFIG_DIR / 'benchmark.yml', encoding='utf-8') as file:
    DESIGN = yaml.safe_load(file)

with open(CONFIG_DIR / 'datasets.yml', encoding='utf-8') as file:
    DATA_SETTINGS = yaml.safe_load(file)

with open(CONFIG_DIR / 'models.yml', encoding='utf-8') as file:
    PANEL = yaml.safe_load(file)

# The scenarios are the one part of the design written by hand, so they are kept
# beside it rather than in a script.
with open(SCENARIOS_PATH, encoding='utf-8') as file:
    SCENARIO_SETTINGS = yaml.safe_load(file)

DOMAINS = DESIGN['domains']
TYPES = DESIGN['types']
SEED = DESIGN['seed']
ANSWERS = DESIGN['answers']
CUES = DESIGN['cues']
SAFETY = DESIGN['safety']
LANGUAGE = DESIGN['language']
METHODS = DESIGN['methods']
PERSISTENCE = DESIGN['persistence']
AGE_BAND_LIMITS = DESIGN['age_bands']
EXPLICIT_OPENER = DESIGN['explicit_opener']

SCENARIOS = SCENARIO_SETTINGS['scenarios']
WHO = SCENARIO_SETTINGS['who']

MODELS = PANEL['models']
JUDGES = PANEL['judges']
JUDGE = JUDGES['primary']
GENERATION = PANEL['generation']
PROVIDER_KEYS = PANEL['keys']

DATASETS = DATA_SETTINGS['datasets']
SOURCES = DATA_SETTINGS['sources']
DATASET_NAMES = sorted({spec['name'] for spec in SOURCES.values()})
DOWNLOAD_NAMES = {name: spec['name'] for name, spec in DATASETS.items()}
SAFECHILD_COMMIT = DATA_SETTINGS['safechild_commit']

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


CONDITIONS = [expand_condition(condition) for condition in DESIGN['conditions']]

# ----------------------------------------------------------------------------
# Derived from the settings
# ----------------------------------------------------------------------------

DOMAIN_NAMES = {code: values['name'] for code, values in DOMAINS.items()}
DOMAIN_CODES = {name: code for code, name in DOMAIN_NAMES.items()}
TYPE_ANSWERS = {name: values['answers'] for name, values in TYPES.items()}
PER_DOMAIN = sum(values['count'] for values in TYPES.values())
TOTAL_SCENARIOS = PER_DOMAIN * len(DOMAINS)

CONDITION_NAMES = [condition['name'] for condition in CONDITIONS]
CONDITION_AGES = [condition['age'] for condition in CONDITIONS]
SIGNALS = sorted({condition['signal'] for condition in CONDITIONS})

KEEP_VALUES = ['yes', '']

# The canonical request and one variant per band. Each variant carries an age
# cue in place of the neutral phrase the canonical request holds. The wording
# runs source_prompt, as the corpus had it, to request, as rewritten here, to
# prompt, as put to a model.
# Define function to name the request variant a band uses
def variant_column(band):
    return f'implicit_{band.lower()}'


VARIANT_COLUMNS = (['request', 'implicit_cue']
                   + [variant_column(band) for band in AGE_BANDS])

DRAFTS_COLUMNS = ['source_id', 'dataset', 'domain', 'scenario_type',
                  'source_prompt', *VARIANT_COLUMNS, 'keep']
BENCHMARK_COLUMNS = ['scenario_id', 'dataset', 'domain', 'scenario_type',
                     *VARIANT_COLUMNS]
PROMPT_COLUMNS = ['prompt_id', 'scenario_id', 'condition', 'age', 'band',
                  'signal', 'cue', 'prompt', 'expected_answer']

# One row per scored reply. The answer is compared against the expectation;
# the safety measures are reported on their own; the language measures are
# computed from the reply text.
JUDGED_COLUMNS = (['prompt_id', 'model', 'replicate', 'answer', 'expected_answer',
                   'matched'] + list(SAFETY) + LANGUAGE)

# Define function to turn a model identifier into a filename. A slash separates
# an owner from a model on the Hugging Face hub and a colon separates a tag in
# Ollama, and neither belongs in a path.
def model_slug(model_id):
    return re.sub(r'[^a-z0-9.-]+', '-', str(model_id).lower()).strip('-')


# Define function to name the file one model writes to in one experiment
def result_path(model_id, directory):
    return directory / f'{model_slug(model_id)}.jsonl'


# Define function to name the file one model's single-turn replies go in
def adaptation_path(model_id):
    return result_path(model_id, ADAPTATION_DIR)


# Define function to name the file one model's age inferences go in
def recognition_path(model_id):
    return result_path(model_id, RECOGNITION_DIR)


# Define function to name the file one model's dialogues go in
def persistence_path(model_id):
    return result_path(model_id, PERSISTENCE_DIR)


# Define function to read every model's replies as one frame
def read_all(directory):
    frames = [read_lines(path) for path in sorted(directory.glob('*.jsonl'))]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Define function to name a column after a measure
def measure_column(name):
    return name.lower().replace(' ', '_')


# One record per reply. The identifying fields come first so that scanning the
# file shows what each line is without reading the reply itself, and the reply
# comes last because it is the only field of unpredictable length.
RESPONSE_FIELDS = ['prompt_id', 'model', 'replicate', 'backend', 'temperature',
                   'error', 'response']

# One row per scored reply: what the model did with the request, the five
# safety measures, and the language measures computed from the text.
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
LEXICAL_COLUMNS = ['scenario_id', 'request', 'mean_aoa', 'max_aoa',
                   'changes', 'n_changes']
SCORE_COLUMNS = ['variant', 'scenario_id', 'words', 'fkgl', 'fre', 'mean_aoa',
                 'max_aoa', 'difficult', 'covered']

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to build a scenario identifier
def make_scenario_id(code, scenario_type, index):
    return f'{code}-{TYPES[scenario_type]["code"]}{index}'


# Define function to build a source identifier
def make_source_id(dataset, record_id):
    return f'{dataset}-{record_id}'


# Define function to read the domain code back out of a scenario identifier
def code_from_scenario(scenario_id):
    return str(scenario_id).split('-')[0]


# Define function to build a prompt identifier
def make_prompt_id(scenario_id, condition):
    return f'{scenario_id}-{condition}'


# Define function to place the age signal in front of the scenario request
def make_prompt(opener, request):
    return f'{opener} {request}'.strip()


# Define function to read the api key a provider expects, from .env
def api_key(provider):
    variable = PROVIDER_KEYS.get(provider)
    if variable is None:
        return ''
    if variable not in os.environ and ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            name, _, value = line.partition('=')
            if name.strip() and not name.strip().startswith('#'):
                os.environ.setdefault(name.strip(), value.strip())
    return os.environ.get(variable, '')


# Define function to describe the shape of a table
def shape_of(frame):
    return f'{len(frame)} rows, {frame.shape[1]} columns'


# Define function to head a block of printed output
_SECTIONS = []


def section(title):
    print(f'\n{title}' if _SECTIONS else title)
    _SECTIONS.append(title)


# Define function to keep only the scenario rows that have been written
def written(scenarios):
    return scenarios[scenarios['request'].str.strip() != ''].reset_index(drop=True)


# Define function to check a table for the faults that break the pipeline
def validate(frame, required, id_column='', text_columns=(), labels=None):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return [f'missing columns {", ".join(missing)}']

    problems = []
    if id_column:
        repeated = frame[id_column][frame[id_column].duplicated()].unique()
        if len(repeated):
            problems.append(f'{len(repeated)} duplicate ids, first {repeated[0]}')
    for column in text_columns:
        blank = frame[column].fillna('').astype(str).str.strip() == ''
        if blank.any():
            problems.append(f'{int(blank.sum())} empty values in {column}')
    for column, allowed in (labels or {}).items():
        values = frame[column].fillna('').astype(str)
        invalid = sorted(set(values) - {str(value) for value in allowed})
        if invalid:
            problems.append(f'invalid {column} labels {", ".join(invalid[:5])}')
    return problems


# Define function to check the drafts a scenario may be selected from
def check_drafts(drafts):
    problems = validate(frame=drafts, required=DRAFTS_COLUMNS,
                        id_column='source_id',
                        text_columns=['source_id', 'domain'],
                        labels={'domain': DOMAIN_NAMES.values(),
                                'dataset': DATASET_NAMES + [''],
                                'implicit_cue': CUES + [''],
                                'scenario_type': TYPES,
                                'keep': KEEP_VALUES})
    if problems:
        return problems
    kept = drafts[drafts['keep'] == 'yes']
    blank = kept['request'].str.strip() == ''
    if blank.any():
        problems.append(f'{int(blank.sum())} kept drafts have no request')

    # every kept scenario carries the cue variants. On age-sensitive scenarios
    # they test whether a cue shifts the answer with the band; on harmful and
    # benign ones, where the expected answer does not vary, they test whether
    # the cue phrase shifts behaviour on its own, which is the control for the
    # change in meaning the phrase also carries
    for column in VARIANT_COLUMNS[1:]:
        missing = kept[column].str.strip() == ''
        if missing.any():
            problems.append(f'{int(missing.sum())} kept drafts have no {column}')
    return problems


# Define function to check the written scenarios
def check_benchmark(scenarios):
    problems = validate(frame=scenarios, required=BENCHMARK_COLUMNS,
                        id_column='scenario_id',
                        text_columns=['scenario_id', 'request'],
                        labels={'scenario_type': TYPES,
                                'dataset': DATASET_NAMES + [''],
                                'implicit_cue': CUES + ['']})
    return problems


# Define function to report validation problems and stop when any are found
def report(name, problems):
    if not problems:
        print(f'Validated {name}')
        return
    for problem in problems:
        print(f'  {name}: {problem}')
    raise SystemExit(f'{len(problems)} validation problems in {name}')


# Define function to make the data directories, so a script can write without
# each one checking first
def make_directories(directories=DATA_DIRS):
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Define function to read a JSON lines file, which is how model output is kept:
# a reply can hold newlines and quotes that CSV quoting mishandles, and a line
# appended as each reply arrives survives a run that stops part way
def read_lines(path):
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


# Define function to append one record to a JSON lines file
def append_line(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as file:
        file.write(json.dumps(record) + '\n')



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
assert all(method in METHODS for method in PERSISTENCE['methods'])
assert all(len(values) >= 2 for values in SAFETY.values())
assert all(len(values) >= 2 for values in SAFETY.values())
assert len({len(spec['turns']) for spec in METHODS.values()}) == 1
assert all(spec['provider'] in PROVIDER_KEYS for spec in MODELS.values()
           if spec['access'] == 'api')
