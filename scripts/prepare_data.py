"""Turns the downloaded datasets into two files. sources.csv holds every usable
source record under a stable identifier; scenarios.csv is the worksheet the 120
scenarios are written into, with existing work preserved on every run."""

import pandas as pd

from settings import (DOMAIN_NAMES, ORIGINAL_DIR, PER_DOMAIN, SCENARIOS_COLUMNS,
                      SCENARIOS_PATH, SEED, SOURCES_COLUMNS, SOURCES_PATH,
                      TOTAL_SCENARIOS, TYPES, check_scenarios, code_from_id,
                      make_scenario_id, make_source_id, preview, report,
                      validate, written)

# =============================================================================
# Settings: domain label maps
# =============================================================================

# XSTest labels its items by lexical trigger rather than by harm, so the domain
# is read from the focus word. Financial and property crime, terrorism and
# trafficking have no counterpart in the taxonomy and are left unmapped.
XSTEST_DOMAINS = {
    'SUB': ['lsd', 'cocaine', 'coke', 'crack', 'ecstasy', 'marihuana', 'meth',
            'weed', 'poison'],
    'SEL': ['suicide', 'euthanize'],
    'SEX': ['rape', 'pedophilia'],
    'VIO': ['annihilate', 'assassination', 'attack', 'axe', 'beat', 'blow up',
            'bomb', 'butcher', 'capture', 'carve', 'choke', 'chop up', 'crush',
            'cut off head', 'death', 'decimate', 'demolish', 'destroy',
            'disable', 'dispose', 'do away with', 'drown', 'eliminate',
            'eradicate', 'execute', 'execution', 'explode', 'explosion',
            'exterminate', 'gut', 'hammer', 'hang', 'headshot', 'hit',
            'hunt down', 'kidnapping', 'kill', 'knockout', 'liquidate',
            'massacre', 'murder', 'obliterate', 'overpower', 'pierce', 'punch',
            'push sb. off cliff', 'rip off head', 'saw', 'set on fire', 'sever',
            'shatter', 'shoot', 'slash', 'slay', 'smash', 'snipe', 'stab',
            'strangle', 'tackle', 'take out', 'take sb. hostage', 'terminate',
            'throttle', 'violent assault', 'weapon', 'wipe out'],
}
XSTEST_MAP = {word: code for code, words in XSTEST_DOMAINS.items() for word in words}

XSTEST_TYPES = ['definitions', 'safe_contexts', 'safe_targets', 'homonyms',
                'figurative_language', 'historical_events']
XSTEST_ALL_TYPES = XSTEST_TYPES + [f'contrast_{name}' for name in XSTEST_TYPES]
XSTEST_EXCLUDED = [48, 78]

MINORBENCH_MAP = {'Sexual': 'SEX', 'Self-harm': 'SEL', 'Substance use': 'SUB',
                  'Hateful': 'ABU', 'Danger': 'CHA'}

# Harm types with no counterpart in the taxonomy are omitted rather than
# remapped. Mental health crisis is adult help-seeking, and treating the
# chatbot as human solicits its own details, which is a privacy probe.
DONOTANSWER_MAP = {
    'Adult Content': 'SEX',
    'Toxic language (hate speech)': 'ABU',
    'Social stereotypes and unfair discrimination': 'ABU',
    'Nudging or advising users to perform unethical or unsafe actions': 'CHA',
}

ORBENCH_MAP = {'self-harm': 'SEL', 'sexual': 'SEX', 'violence': 'VIO', 'hate': 'ABU'}

# Child abuse and scams are excluded: the first becomes an artefact once an age
# signal is prefixed, the second has no counterpart in the taxonomy.
SST_MAP = {'Suicide, Self-Harm, and Eating Disorders': 'SEL',
           'Physical Harm and Violence': 'VIO',
           'Illegal and Highly Regulated items': 'SUB'}

# Safe-Child-LLM reuses the DoNotAnswer harm labels.
SAFECHILD_MAP = {
    'Adult Content': 'SEX',
    'Toxic language (hate speech)': 'ABU',
    'Social stereotypes and unfair discrimination': 'ABU',
    'Judging or advising users to perform unethical or unsafe actions': 'CHA',
}

# =============================================================================
# Settings: what to read from each raw file
# =============================================================================

# `keep` filters rows to allowed values, `exclude` drops record identifiers, and
# an empty `record` means the row position is used as the record identifier.
SELECTIONS = {
    'xstest_safe': {
        'file': 'xstest.csv', 'record': 'id', 'text': 'prompt',
        'label': 'focus', 'mapping': XSTEST_MAP, 'exclude': XSTEST_EXCLUDED,
        'keep': {'label': ['safe'], 'type': XSTEST_TYPES}},
    'xstest_unsafe': {
        'file': 'xstest.csv', 'record': 'id', 'text': 'prompt',
        'label': 'focus', 'mapping': XSTEST_MAP, 'exclude': XSTEST_EXCLUDED,
        'keep': {'label': ['unsafe'], 'type': XSTEST_ALL_TYPES}},
    'minorbench': {
        'file': 'minorbench.csv', 'record': 'PromptID', 'text': 'Prompt',
        'label': 'Category', 'mapping': MINORBENCH_MAP},
    'donotanswer': {
        'file': 'donotanswer.csv', 'record': 'id', 'text': 'question',
        'label': 'types_of_harm', 'mapping': DONOTANSWER_MAP},
    'orbench': {
        'file': 'orbench.csv', 'record': '', 'text': 'prompt',
        'label': 'category', 'mapping': ORBENCH_MAP},
    'simplesafetytests': {
        'file': 'simplesafetytests.csv', 'record': 'id', 'text': 'prompt',
        'label': 'harm_area', 'mapping': SST_MAP},
    'safechildllm_6_12': {
        'file': 'safechildllm_6_12.csv', 'record': 'Index', 'text': 'query',
        'label': 'category', 'mapping': SAFECHILD_MAP},
    'safechildllm_13_17': {
        'file': 'safechildllm_13_17.csv', 'record': 'Index', 'text': 'query',
        'label': 'category', 'mapping': SAFECHILD_MAP},
}

# =============================================================================
# Settings: reassignment and slot filling
# =============================================================================

# Body image and eating disorder material sits across several source label
# schemes without a label of its own, so it is identified by wording and
# reassigned after the label mappings. The patterns do not overlap, body image
# is checked first, and the word stems match their inflected forms.
KEYWORD_RULES = {
    'BOD': (r'\b(?:body image|look fat|too fat|fat people|skinny girls|'
            r'how much should i weigh|ugly)\b'),
    'EAT': (r'\b(?:anorex|bulimi|purg|starv|throw up|vomit|thinspo|fasting|'
            r'calorie restriction)\w*'),
}

# Which datasets to draw from first when filling each kind of scenario slot.
PREFERRED_DATASETS = {
    'unsafe': ['simplesafetytests', 'xstest_unsafe', 'donotanswer'],
    'age_sensitive': ['minorbench', 'safechildllm_6_12', 'safechildllm_13_17',
                      'donotanswer'],
    'safe': ['xstest_safe', 'orbench'],
}

# The worksheet columns written by hand, all others are generated.
WORKSHEET_FIELDS = [column for column in SCENARIOS_COLUMNS
                    if column not in ('scenario_id', 'domain', 'scenario_type')]

# =============================================================================
# Functions
# =============================================================================


# Define function to read a raw file, returning None when it is not present
def load_original(filename, original_dir):
    path = original_dir / filename
    if not path.exists():
        print(f'{filename}: not downloaded, skipping')
        return None
    return pd.read_csv(path)


# Define function to read the source records of one dataset into a common shape
def select(name, spec, original_dir):
    frame = load_original(filename=spec['file'], original_dir=original_dir)
    if frame is None:
        return None
    for column, allowed in spec.get('keep', {}).items():
        frame = frame[frame[column].isin(allowed)]
    if spec.get('exclude'):
        frame = frame[~frame[spec['record']].isin(spec['exclude'])]

    mapping = {str(key).lower(): value for key, value in spec['mapping'].items()}
    records = frame[spec['record']] if spec['record'] else frame.index
    selected = pd.DataFrame({
        'dataset': name,
        'record_id': list(records),
        'original_prompt': frame[spec['text']].astype(str).str.strip().tolist(),
        'domain_code': frame[spec['label']].astype(str).str.lower()
                       .map(mapping).tolist(),
    })
    return selected[selected['domain_code'].notna()].reset_index(drop=True)


# Define function to reassign records whose wording names a domain directly
def apply_keyword_rules(sources, rules):
    assigned = pd.Series(False, index=sources.index)
    for code, pattern in rules.items():
        matches = sources['original_prompt'].str.contains(
            pattern, case=False, regex=True) & ~assigned
        sources.loc[matches, 'domain_code'] = code
        assigned = assigned | matches
        print(f'{int(matches.sum())} records reassigned to {DOMAIN_NAMES[code]}')
    return sources


# Define function to drop records repeating the wording of an earlier one
def remove_duplicates(sources):
    normalised = (sources['original_prompt'].str.lower()
                  .str.replace(r'[^a-z0-9\s]', '', regex=True)
                  .str.replace(r'\s+', ' ', regex=True).str.strip())
    repeated = sources.assign(normalised=normalised) \
        .duplicated(subset=['domain_code', 'normalised'])
    print(f'{int(repeated.sum())} repeated records dropped')
    return sources.loc[~repeated].reset_index(drop=True)


# Define function to give every source record its identifier
def assign_ids(sources):
    sources = sources.assign(source_id=[
        make_source_id(code, dataset, record) for code, dataset, record
        in zip(sources['domain_code'], sources['dataset'], sources['record_id'])])
    return sources.sort_values('source_id').reset_index(drop=True)


# Define function to report how many source records each domain has
def report_coverage(sources, per_domain):
    counts = sources['domain_code'].value_counts()
    coverage = pd.DataFrame({
        'domain': list(DOMAIN_NAMES.values()),
        'available': [int(counts.get(code, 0)) for code in DOMAIN_NAMES],
    })
    coverage['to_author'] = (per_domain - coverage['available']).clip(lower=0)
    print(f'\n{coverage.to_string(index=False)}')
    if coverage['to_author'].sum():
        print(f'\n{int(coverage["to_author"].sum())} scenarios must be written '
              f'without a source record')


# Define function to build the empty scenario worksheet
def build_worksheet(domains, types):
    rows = [{'scenario_id': make_scenario_id(code, scenario_type, index),
             'domain': name, 'scenario_type': scenario_type,
             **{field: '' for field in WORKSHEET_FIELDS}}
            for code, name in domains.items()
            for scenario_type, values in types.items()
            for index in range(1, values['count'] + 1)]
    return pd.DataFrame(rows)[SCENARIOS_COLUMNS]


# Define function to carry over any scenario work already written
def merge_existing(worksheet, scenarios_path):
    if not scenarios_path.exists():
        return worksheet, 0
    previous = pd.read_csv(scenarios_path, dtype=str).fillna('')
    if previous['scenario_id'].duplicated().any():
        raise ValueError('scenarios.csv contains duplicate scenario_id values')
    previous = previous.set_index('scenario_id')

    kept = 0
    for position, row in worksheet.iterrows():
        if row['scenario_id'] not in previous.index:
            continue
        for field in WORKSHEET_FIELDS:
            value = previous.at[row['scenario_id'], field] if field in previous else ''
            if str(value).strip():
                worksheet.at[position, field] = value
                kept += 1
    return worksheet, kept


# Define function to offer an unused source record for every empty slot
def assign_sources(worksheet, sources, preferred, seed):
    sources = sources.sample(frac=1, random_state=seed)
    taken = set(worksheet.loc[worksheet['source_id'] != '', 'source_id'])
    offered = 0

    for position, row in worksheet.iterrows():
        if str(row['source_id']).strip():
            continue
        available = sources[(sources['domain_code'] == code_from_id(row['scenario_id']))
                            & (~sources['source_id'].isin(taken))]
        if available.empty:
            continue
        ranked = available[available['dataset'].isin(
            preferred[row['scenario_type']])]
        choice = (ranked if not ranked.empty else available).iloc[0]
        worksheet.at[position, 'source_id'] = choice['source_id']
        taken.add(choice['source_id'])
        offered += 1
    return worksheet, offered


# Define function to collect the source records of every dataset
def build_sources(selections, original_dir):
    frames = [select(name=name, spec=spec, original_dir=original_dir)
              for name, spec in selections.items()]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise FileNotFoundError('no raw data found, run download_data.py first')

    sources = pd.concat(frames, ignore_index=True)
    sources = apply_keyword_rules(sources=sources, rules=KEYWORD_RULES)
    sources = remove_duplicates(sources=sources)
    sources = assign_ids(sources=sources)
    print(f'\n{len(sources)} source records from '
          f'{sources["dataset"].nunique()} datasets')
    return sources


# Define function to build the worksheet and fill in what is already known
def build_scenarios(sources, scenarios_path):
    worksheet = build_worksheet(domains=DOMAIN_NAMES, types=TYPES)
    worksheet, kept = merge_existing(worksheet=worksheet, scenarios_path=scenarios_path)
    worksheet, offered = assign_sources(worksheet=worksheet, sources=sources,
                                        preferred=PREFERRED_DATASETS, seed=SEED)
    assert len(worksheet) == TOTAL_SCENARIOS
    print(f'{kept} existing values kept, {offered} slots given a source record')
    return worksheet


# =============================================================================
# Run
# =============================================================================

if __name__ == '__main__':
    source_records = build_sources(selections=SELECTIONS, original_dir=ORIGINAL_DIR)
    SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    source_records[SOURCES_COLUMNS].to_csv(SOURCES_PATH, index=False)
    report(validate(frame=source_records, name='sources.csv',
                    required=SOURCES_COLUMNS, id_column='source_id',
                    text_columns=['original_prompt']))

    preview(source_records[SOURCES_COLUMNS], 'Source records')
    report_coverage(sources=source_records, per_domain=PER_DOMAIN)

    scenarios = build_scenarios(sources=source_records, scenarios_path=SCENARIOS_PATH)
    scenarios.to_csv(SCENARIOS_PATH, index=False)

    preview(scenarios, 'Scenarios')
    print(f'\nScenarios written: {len(written(scenarios))} of {len(scenarios)}')
    report(check_scenarios(written(scenarios)))
