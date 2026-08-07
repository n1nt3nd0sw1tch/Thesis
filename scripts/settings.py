"""Loads the benchmark design and the dataset settings, then provides the
helpers the scripts share: identifier construction, the scenario type rule, and
validation."""

from pathlib import Path
import yaml

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'config'
DATA_DIR = ROOT / 'data'
ORIGINAL_DIR = DATA_DIR / 'original'
SOURCES_PATH = DATA_DIR / 'sources.csv'
DRAFTS_PATH = DATA_DIR / 'drafts.csv'
SCENARIOS_PATH = DATA_DIR / 'scenarios.csv'
PROMPTS_PATH = DATA_DIR / 'prompts.csv'
DATASETS_DOC_PATH = DATA_DIR / 'datasets.md'

# ----------------------------------------------------------------------------
# Settings, read from config/
# ----------------------------------------------------------------------------

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

# An omitted age, band or opener reads as None from yaml and is held as an
# empty string so that the age column stays whole numbers once written to csv.
CONDITIONS = [{**condition,
               **{field: '' if condition.get(field) is None else condition[field]
                  for field in ('age', 'band', 'opener')}}
              for condition in BENCHMARK['conditions']]

# ----------------------------------------------------------------------------
# Derived from the settings
# ----------------------------------------------------------------------------

DOMAIN_NAMES = {code: values['name'] for code, values in DOMAINS.items()}
AGE_BANDS = list(AGE_BAND_LIMITS)
ACTION_LEVEL = {action: level for level, action in enumerate(ACTIONS)}
TYPE_ACTIONS = {name: values['actions'] for name, values in TYPES.items()}
PER_DOMAIN = sum(values['count'] for values in TYPES.values())
TOTAL_SCENARIOS = PER_DOMAIN * len(DOMAINS)

CONDITION_NAMES = [condition['name'] for condition in CONDITIONS]
CONDITION_AGES = [condition['age'] for condition in CONDITIONS]
SIGNALS = sorted({condition['signal'] for condition in CONDITIONS})
CUES = sorted({condition['cue'] for condition in CONDITIONS})

KEEP_VALUES = ['yes', '']

SOURCES_COLUMNS = ['source_id', 'dataset', 'record_id', 'original_request']
DRAFTS_COLUMNS = ['source_id', 'domain', 'scenario_type', 'original_request',
                  'request', 'keep']
SCENARIOS_COLUMNS = ['scenario_id', 'source_id', 'domain', 'scenario_type',
                     'original_request', 'request', *AGE_BANDS]
PROMPT_COLUMNS = ['prompt_id', 'scenario_id', 'condition', 'signal', 'band',
                  'age', 'prompt', 'expected_action']

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

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
def make_prompt_id(scenario_id, condition):
    return f'{scenario_id}-{condition}'


# Define function to place the age signal in front of the scenario request
def make_prompt(opener, request):
    return f'{opener} {request}'.strip()


# Define function to read the domain code back out of any identifier
def code_from_id(identifier):
    return str(identifier).split('-')[0]


# Define function to describe the shape of a table
def shape_of(frame):
    return f'{len(frame)} rows, {frame.shape[1]} columns'


# Define function to head a block of printed output
def section(title):
    print(f'\n{title}')


# Define function to name the type a set of three expected actions belongs to
def derive_type(child, adolescent, adult):
    actions = [child, adolescent, adult]
    unknown = [action for action in actions if action not in ACTIONS]
    if unknown:
        raise ValueError(f'unrecognised action: {unknown}')
    for name, expected in TYPE_ACTIONS.items():
        if expected == actions:
            return name
    raise ValueError(f'{", ".join(actions)} is not an allowed set of actions')


# Define function to keep only the scenario rows that have been written
def written(scenarios):
    return scenarios[scenarios['request'].str.strip() != ''].reset_index(drop=True)


# Define function to check the expected actions of one written scenario
def check_actions(scenario):
    actions = [scenario[band] for band in AGE_BANDS]
    if not all(str(action).strip() for action in actions):
        return f'{scenario["scenario_id"]} has incomplete expected actions'
    try:
        derived = derive_type(*actions)
    except ValueError as error:
        return f'{scenario["scenario_id"]} {error}'
    if derived != scenario['scenario_type']:
        return (f'{scenario["scenario_id"]} actions imply {derived}, '
                f'slot is {scenario["scenario_type"]}')
    return ''


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


# Define function to check the source records
def check_sources(sources):
    return validate(frame=sources, required=SOURCES_COLUMNS,
                    id_column='source_id', text_columns=['original_request'])


# Define function to check the drafts a scenario may be selected from
def check_drafts(drafts):
    problems = validate(frame=drafts, required=DRAFTS_COLUMNS,
                        id_column='source_id',
                        text_columns=['source_id', 'domain'],
                        labels={'scenario_type': TYPES, 'keep': KEEP_VALUES})
    if problems:
        return problems
    kept = drafts[drafts['keep'] == 'yes']
    blank = kept['request'].str.strip() == ''
    if blank.any():
        problems.append(f'{int(blank.sum())} kept drafts have no request')
    return problems


# Define function to check the written scenarios
def check_scenarios(scenarios):
    problems = validate(frame=scenarios, required=SCENARIOS_COLUMNS,
                        id_column='scenario_id',
                        text_columns=['scenario_id', 'request'],
                        labels={'scenario_type': TYPES,
                                **{band: ACTIONS for band in AGE_BANDS}})
    if problems:
        return problems
    return [problem for _, scenario in scenarios.iterrows()
            if (problem := check_actions(scenario))]


# Define function to report validation problems and stop when any are found
def report(name, problems):
    if not problems:
        print(f'Validated {name}')
        return
    for problem in problems:
        print(f'  {name}: {problem}')
    raise SystemExit(f'{len(problems)} validation problems in {name}')

# ----------------------------------------------------------------------------
# Checks applied to the settings when this module is imported
# ----------------------------------------------------------------------------

assert all(condition['band'] == band_for_age(condition['age'])
           for condition in CONDITIONS if condition['age'] != '')
assert len(set(CONDITION_NAMES)) == len(CONDITIONS)
assert set(AGE_BANDS) >= {condition['band'] for condition in CONDITIONS
                          if condition['band']}
assert all(len(actions) == len(AGE_BANDS) for actions in TYPE_ACTIONS.values())
assert all(action in ACTIONS for actions in TYPE_ACTIONS.values()
           for action in actions)
assert all([ACTION_LEVEL[action] for action in actions]
           == sorted(ACTION_LEVEL[action] for action in actions)
           for actions in TYPE_ACTIONS.values())
assert len({tuple(actions) for actions in TYPE_ACTIONS.values()}) == len(TYPES)
assert all({'kind', 'licence'} <= set(spec) for spec in DATASETS.values())
assert all('origin' in spec if spec['kind'] == 'hub' else 'filename' in spec
           for spec in DATASETS.values())
