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
DOWNLOADS_PATH = DATA_DIR / 'downloads.md'
DRAFTS_PATH = DATA_DIR / 'drafts.csv'
BENCHMARK_PATH = DATA_DIR / 'benchmark.csv'
LEXICAL_PATH = DATA_DIR / 'lexical.csv'
SCORES_PATH = DATA_DIR / 'scores.csv'
PROMPTS_PATH = DATA_DIR / 'prompts.csv'

# ----------------------------------------------------------------------------
# Settings, read from config/
# ----------------------------------------------------------------------------

with open(CONFIG_DIR / 'benchmark.yml', encoding='utf-8') as file:
    DESIGN = yaml.safe_load(file)

with open(CONFIG_DIR / 'datasets.yml', encoding='utf-8') as file:
    DATA_SETTINGS = yaml.safe_load(file)

DOMAINS = DESIGN['domains']
TYPES = DESIGN['types']
ACTIONS = DESIGN['actions']
CUES = DESIGN['cues']
AGE_BAND_LIMITS = DESIGN['age_bands']

DATASETS = DATA_SETTINGS['datasets']
SOURCES = DATA_SETTINGS['sources']
SAFECHILD_COMMIT = DATA_SETTINGS['safechild_commit']

# An omitted age, band or opener reads as None from yaml and is held as an
# empty string so that the age column stays whole numbers once written to csv.
CONDITIONS = [{**condition,
               **{field: '' if condition.get(field) is None else condition[field]
                  for field in ('age', 'band', 'request', 'opener')}}
              for condition in DESIGN['conditions']]

# ----------------------------------------------------------------------------
# Derived from the settings
# ----------------------------------------------------------------------------

DOMAIN_NAMES = {code: values['name'] for code, values in DOMAINS.items()}
AGE_BANDS = list(AGE_BAND_LIMITS)
TYPE_ACTIONS = {name: values['actions'] for name, values in TYPES.items()}
PER_DOMAIN = sum(values['count'] for values in TYPES.values())
TOTAL_SCENARIOS = PER_DOMAIN * len(DOMAINS)

CONDITION_NAMES = [condition['name'] for condition in CONDITIONS]
CONDITION_AGES = [condition['age'] for condition in CONDITIONS]
SIGNALS = sorted({condition['signal'] for condition in CONDITIONS})

KEEP_VALUES = ['yes', '']

# The neutral prompt and one variant per band. Each variant carries an age cue
# in place of the neutral phrase the prompt holds.
VARIANT_COLUMNS = (['prompt', 'implicit_cue']
                   + [f'implicit_{band}' for band in AGE_BANDS])

DRAFTS_COLUMNS = ['source_id', 'domain', 'scenario_type', 'original_prompt',
                  *VARIANT_COLUMNS, 'keep']
BENCHMARK_COLUMNS = ['scenario_id', 'source_id', 'domain', 'scenario_type',
                     *VARIANT_COLUMNS]
PROMPT_COLUMNS = ['prompt_id', 'scenario_id', 'condition', 'age', 'band',
                  'signal', 'cue', 'prompt', 'expected_action']
LEXICAL_COLUMNS = ['scenario_id', 'request', 'mean_aoa', 'max_aoa',
                   'changes', 'n_changes']
SCORE_COLUMNS = ['variant', 'scenario_id', 'words', 'fkgl', 'fre', 'mean_aoa',
                 'max_aoa', 'difficult', 'covered']

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
def make_source_id(dataset, record_id):
    return f'{dataset}-{record_id}'


# Define function to build a prompt identifier
def make_prompt_id(scenario_id, condition):
    return f'{scenario_id}-{condition}'


# Define function to place the age signal in front of the scenario request
def make_prompt(opener, request):
    return f'{opener} {request}'.strip()


# Define function to describe the shape of a table
def shape_of(frame):
    return f'{len(frame)} rows, {frame.shape[1]} columns'


# Define function to head a block of printed output
def section(title):
    print(f'\n{title}')


# Define function to keep only the scenario rows that have been written
def written(scenarios):
    return scenarios[scenarios['prompt'].str.strip() != ''].reset_index(drop=True)


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
                                'implicit_cue': CUES + [''],
                                'scenario_type': TYPES,
                                'keep': KEEP_VALUES})
    if problems:
        return problems
    kept = drafts[drafts['keep'] == 'yes']
    blank = kept['prompt'].str.strip() == ''
    if blank.any():
        problems.append(f'{int(blank.sum())} kept drafts have no prompt')

    # an embedded cue can only show an effect where the action varies by age,
    # so the variants are required on age-sensitive scenarios and optional
    # elsewhere
    sensitive = kept[kept['scenario_type'] == 'age_sensitive']
    for column in VARIANT_COLUMNS[1:]:
        missing = sensitive[column].str.strip() == ''
        if missing.any():
            problems.append(f'{int(missing.sum())} age-sensitive drafts have '
                            f'no {column}')
    return problems


# Define function to check the written scenarios
def check_benchmark(scenarios):
    problems = validate(frame=scenarios, required=BENCHMARK_COLUMNS,
                        id_column='scenario_id',
                        text_columns=['scenario_id', 'prompt'],
                        labels={'scenario_type': TYPES,
                                'implicit_cue': CUES + ['']})


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
assert all([ACTIONS.index(action) for action in actions]
           == sorted(ACTIONS.index(action) for action in actions)
           for actions in TYPE_ACTIONS.values())
assert len({tuple(actions) for actions in TYPE_ACTIONS.values()}) == len(TYPES)
assert all({'kind', 'licence'} <= set(spec) for spec in DATASETS.values())
assert all({'file', 'scenario_type', 'text', 'label', 'domains'} <= set(spec)
           for spec in SOURCES.values())
assert all(spec['scenario_type'] in TYPES for spec in SOURCES.values())
assert all(code in DOMAINS for spec in SOURCES.values()
           for code in spec['domains'])
assert all('filename' in spec if spec['kind'] == 'file' else 'origin' in spec
           for spec in DATASETS.values())
assert all(spec['kind'] in ('hub', 'url', 'file') for spec in DATASETS.values())
