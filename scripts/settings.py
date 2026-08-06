"""Loads the benchmark design and the dataset settings, then provides the
helpers the scripts share: identifier construction, the scenario type rule, and
validation. Settings live in config/, program behaviour lives here."""

from pathlib import Path

import yaml

# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'config'
DATA_DIR = ROOT / 'data'
ORIGINAL_DIR = DATA_DIR / 'original'
SOURCES_PATH = DATA_DIR / 'sources.csv'
SCENARIOS_PATH = DATA_DIR / 'scenarios.csv'
PROMPTS_PATH = DATA_DIR / 'prompts.csv'
DATASETS_DOC_PATH = DATA_DIR / 'datasets.md'

# =============================================================================
# Settings, read from config/
# =============================================================================

with open(CONFIG_DIR / 'benchmark.yml', encoding='utf-8') as file:
    BENCHMARK = yaml.safe_load(file)

with open(CONFIG_DIR / 'datasets.yml', encoding='utf-8') as file:
    DATA_SETTINGS = yaml.safe_load(file)

SEED = BENCHMARK['seed']
DOMAINS = BENCHMARK['domains']
TYPES = BENCHMARK['types']
ACTIONS = BENCHMARK['actions']
AGE_BAND_LIMITS = BENCHMARK['age_bands']

DATASETS = DATA_SETTINGS['datasets']
SAFECHILD_COMMIT = DATA_SETTINGS['safechild_commit']

# An omitted age or band reads as None from yaml and is held as an empty string
# so that the age column stays whole numbers once written to csv.
CONDITIONS = [{**condition,
               'age': '' if condition.get('age') is None else condition['age'],
               'band': '' if condition.get('band') is None else condition['band']}
              for condition in BENCHMARK['conditions']]

# =============================================================================
# Derived from the settings
# =============================================================================

DOMAIN_NAMES = {code: values['name'] for code, values in DOMAINS.items()}
AGE_BANDS = list(AGE_BAND_LIMITS)
ACTION_LEVEL = {action: level for level, action in enumerate(ACTIONS)}
PER_DOMAIN = sum(values['count'] for values in TYPES.values())
TOTAL_SCENARIOS = PER_DOMAIN * len(DOMAINS)

TURNS = [1, 2]
CONDITION_NAMES = [condition['name'] for condition in CONDITIONS]
CONDITION_AGES = [condition['age'] for condition in CONDITIONS]
SIGNALS = sorted({condition['signal'] for condition in CONDITIONS})

SOURCES_COLUMNS = ['source_id', 'dataset', 'record_id', 'original_prompt']
SCENARIOS_COLUMNS = ['scenario_id', 'source_id', 'domain', 'scenario_type',
                     'request', *AGE_BANDS]
PROMPT_COLUMNS = ['prompt_id', 'scenario_id', 'condition', 'age', 'signal',
                  'turn', 'prompt', 'expected_action']

# =============================================================================
# Functions
# =============================================================================


# Define function to place an age in its band
def band_for_age(age):
    if age == '' or age is None:
        return ''
    age = int(age)
    for band, limits in AGE_BAND_LIMITS.items():
        upper = limits.get('max_age')
        if age >= limits['min_age'] and (upper is None or age <= upper):
            return band
    raise ValueError(f'age {age} falls outside every band')


# Define function to build a scenario identifier
def make_scenario_id(code, scenario_type, index):
    return f'{code}-{TYPES[scenario_type]["code"]}{index}'


# Define function to build a source identifier
def make_source_id(code, dataset, record_id):
    return f'{code}-{dataset}-{record_id}'


# Define function to build a prompt identifier
def make_prompt_id(scenario_id, condition, turn):
    return f'{scenario_id}-{condition}-t{turn}'


# Define function to read the domain code back out of any identifier
def code_from_id(identifier):
    return str(identifier).split('-')[0]


# Define function to derive the scenario type from its three expected actions
def derive_type(child, adolescent, adult):
    actions = [child, adolescent, adult]
    unknown = [action for action in actions if action not in ACTIONS]
    if unknown:
        raise ValueError(f'unrecognised action: {unknown}')
    if all(action == ACTIONS[0] for action in actions):
        return 'unsafe'
    if all(action == ACTIONS[-1] for action in actions):
        return 'safe'
    if len(set(actions)) == 1:
        raise ValueError('actions do not vary by age')
    levels = [ACTION_LEVEL[action] for action in actions]
    if levels != sorted(levels):
        raise ValueError('actions must not become more restrictive with age')
    return 'age_sensitive'


# Define function to keep only the scenario rows that have been written
def written(scenarios):
    return scenarios[scenarios['request'].str.strip() != ''].reset_index(drop=True)


# Define function to check the expected actions of one written scenario
def check_actions(scenario):
    actions = [scenario[band] for band in AGE_BANDS]
    if not all(str(action).strip() for action in actions):
        return f'{scenario["scenario_id"]}: expected actions incomplete'
    try:
        derived = derive_type(*actions)
    except ValueError as error:
        return f'{scenario["scenario_id"]}: {error}'
    if derived != scenario['scenario_type']:
        return (f'{scenario["scenario_id"]}: actions imply {derived}, '
                f'slot is {scenario["scenario_type"]}')
    return ''


# Define function to check a table for the faults that break the pipeline
def validate(frame, name, required, id_column='', text_columns=(), labels=None):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return [f'{name}: missing columns {", ".join(missing)}']

    problems = []
    if id_column:
        repeated = frame[id_column][frame[id_column].duplicated()].unique()
        if len(repeated):
            problems.append(f'{name}: {len(repeated)} duplicate ids, '
                            f'first {repeated[0]}')
    for column in text_columns:
        blank = frame[column].fillna('').astype(str).str.strip() == ''
        if blank.any():
            problems.append(f'{name}: {int(blank.sum())} empty values in {column}')
    for column, allowed in (labels or {}).items():
        values = frame[column].fillna('').astype(str)
        invalid = sorted(set(values) - {str(value) for value in allowed})
        if invalid:
            problems.append(f'{name}: invalid {column} labels '
                            f'{", ".join(invalid[:5])}')
    return problems


# Define function to check the written scenarios as a whole
def check_scenarios(scenarios):
    problems = validate(frame=scenarios, name='scenarios.csv',
                        required=SCENARIOS_COLUMNS, id_column='scenario_id',
                        text_columns=['scenario_id', 'request'],
                        labels={'scenario_type': TYPES,
                                **{band: ACTIONS for band in AGE_BANDS}})
    if problems:
        return problems
    return [problem for _, scenario in scenarios.iterrows()
            if (problem := check_actions(scenario))]


# Define function to show the head of a table and how many rows it holds
def preview(frame, name, rows=5):
    print(f'\n{name}:')
    print(frame.head(rows).to_string(index=False))
    print(f'{name} size: {len(frame)}')


# Define function to print validation problems and stop when any are found
def report(problems):
    if not problems:
        print('validation passed')
        return
    for problem in problems:
        print(f'  {problem}')
    raise SystemExit(f'{len(problems)} validation problems')


# =============================================================================
# Checks applied to the settings when this module is imported
# =============================================================================

assert all(condition['band'] == band_for_age(condition['age'])
           for condition in CONDITIONS if condition['age'] != '')
assert len(set(CONDITION_NAMES)) == len(CONDITIONS)
assert set(AGE_BANDS) >= {condition['band'] for condition in CONDITIONS
                          if condition['band']}
assert all({'kind', 'licence'} <= set(spec) for spec in DATASETS.values())
assert all('origin' in spec if spec['kind'] == 'hub' else 'filename' in spec
           for spec in DATASETS.values())
