"""Turns the downloaded datasets into three files. sources.csv holds every usable
source record under a stable identifier, drafts.csv is the worksheet every
scenario is written and marked in, and scenarios.csv is the 120 slots filled from
the drafts marked to keep. Work in drafts.csv is preserved on every run."""

import pandas as pd
from settings import (AGE_BANDS, DOMAIN_NAMES, DRAFTS_COLUMNS, DRAFTS_PATH,
                      ORIGINAL_DIR, PER_DOMAIN, SCENARIOS_COLUMNS,
                      SCENARIOS_PATH, SOURCES_COLUMNS, SOURCES_PATH,
                      TOTAL_SCENARIOS, TYPE_ACTIONS, TYPES, check_drafts,
                      check_scenarios, check_sources, code_from_id,
                      make_scenario_id, make_source_id, report, section,
                      shape_of, written)

# ----------------------------------------------------------------------------
# Settings: domain label maps
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# Settings: what to read from each raw file
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# Settings: reassignment and slot filling
# ----------------------------------------------------------------------------

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

# The scenario type each dataset is proposed for. The proposal is a starting
# point only: the type is edited in drafts.csv wherever a record suits another.
DATASET_TYPES = {
    'simplesafetytests': 'unsafe',
    'xstest_unsafe': 'unsafe',
    'donotanswer': 'unsafe',
    'minorbench': 'age_sensitive',
    'safechildllm_6_12': 'age_sensitive',
    'safechildllm_13_17': 'age_sensitive',
    'xstest_safe': 'safe',
    'orbench': 'safe',
}

# The draft columns written by hand, all others are generated on every run.
DRAFT_FIELDS = ['scenario_type', 'request', 'keep']

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to read a raw file, returning None when it is not present
def load_original(filename, original_dir):
    path = original_dir / filename
    if not path.exists():
        print(f'Skipped {filename}, not downloaded')
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
        'original_request': frame[spec['text']].astype(str).str.strip().tolist(),
        'domain_code': frame[spec['label']].astype(str).str.lower()
                       .map(mapping).tolist(),
    })
    return selected[selected['domain_code'].notna()].reset_index(drop=True)


# Define function to reassign records whose wording names a domain directly
def apply_keyword_rules(sources, rules):
    assigned = pd.Series(False, index=sources.index)
    for code, pattern in rules.items():
        matches = sources['original_request'].str.contains(
            pattern, case=False, regex=True) & ~assigned
        sources.loc[matches, 'domain_code'] = code
        assigned = assigned | matches
        print(f'Reassigned {int(matches.sum())} records to {DOMAIN_NAMES[code]}')
    return sources


# Define function to drop records repeating the wording of an earlier one
def remove_duplicates(sources):
    normalised = (sources['original_request'].str.lower()
                  .str.replace(r'[^a-z0-9\s]', '', regex=True)
                  .str.replace(r'\s+', ' ', regex=True).str.strip())
    repeated = sources.assign(normalised=normalised) \
        .duplicated(subset=['domain_code', 'normalised'])
    print(f'Removed {int(repeated.sum())} repeated records')
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
    section('Coverage by domain')
    print(coverage.to_string(index=False))
    if coverage['to_author'].sum():
        print(f'Unsourced: {int(coverage["to_author"].sum())} scenarios to write '
              f'without a source record')


# Define function to open a draft for every source record
def build_drafts(sources):
    return pd.DataFrame({
        'source_id': sources['source_id'],
        'domain': sources['domain_code'].map(DOMAIN_NAMES),
        'scenario_type': sources['dataset'].map(DATASET_TYPES),
        'original_request': sources['original_request'],
        'request': '',
        'keep': '',
    })[DRAFTS_COLUMNS]


# Define function to carry over the drafts already written and marked
def merge_drafts(drafts, drafts_path):
    if not drafts_path.exists():
        return drafts, 0
    previous = pd.read_csv(drafts_path, dtype=str).fillna('')
    if previous['source_id'].duplicated().any():
        raise ValueError('drafts.csv contains duplicate source_id values')

    authored = previous[~previous['source_id'].isin(drafts['source_id'])]
    drafts = drafts.set_index('source_id')
    written_before = previous.set_index('source_id')
    kept = 0
    for source_id in drafts.index.intersection(written_before.index):
        for field in DRAFT_FIELDS:
            value = written_before.at[source_id, field]
            if str(value).strip():
                drafts.at[source_id, field] = value
                kept += 1
    drafts = pd.concat([drafts.reset_index(), authored], ignore_index=True)
    return drafts[DRAFTS_COLUMNS], kept


# Define function to fill the scenario slots from the drafts marked to keep
def build_scenarios(drafts, domains, types):
    kept = drafts[drafts['keep'].str.strip().str.lower() == 'yes']
    rows = []
    for code, name in domains.items():
        for scenario_type, values in types.items():
            chosen = kept[(kept['source_id'].map(code_from_id) == code)
                          & (kept['scenario_type'] == scenario_type)]
            for index in range(1, values['count'] + 1):
                draft = chosen.iloc[index - 1] if index <= len(chosen) else None
                rows.append({
                    'scenario_id': make_scenario_id(code, scenario_type, index),
                    'source_id': draft['source_id'] if draft is not None else '',
                    'domain': name,
                    'scenario_type': scenario_type,
                    'original_request': draft['original_request'] if draft is not None else '',
                    'request': draft['request'] if draft is not None else '',
                    **dict(zip(AGE_BANDS, TYPE_ACTIONS[scenario_type])),
                })
    return pd.DataFrame(rows)[SCENARIOS_COLUMNS]


# Define function to report where too few or too many drafts have been kept
def report_selection(drafts, domains, types):
    kept = drafts[drafts['keep'].str.strip().str.lower() == 'yes']
    counts = kept.groupby(['domain', 'scenario_type']).size()
    selection = pd.DataFrame(
        [[int(counts.get((name, scenario_type), 0)) - values['count']
          for scenario_type, values in types.items()] for name in domains.values()],
        index=list(domains.values()), columns=list(types)).rename_axis('domain')
    section('Drafts kept against each slot')
    print(selection.to_string())
    short = int(selection.clip(upper=0).abs().to_numpy().sum())
    spare = int(selection.clip(lower=0).to_numpy().sum())
    print(f'Short: {short} slots have no draft, Spare: {spare} kept drafts unused')



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
    print(f'Sources: {shape_of(sources[SOURCES_COLUMNS])} '
          f'from {sources["dataset"].nunique()} datasets')
    return sources


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    section('Source records')
    source_records = build_sources(selections=SELECTIONS, original_dir=ORIGINAL_DIR)
    SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    source_records[SOURCES_COLUMNS].to_csv(SOURCES_PATH, index=False)
    report('sources.csv', check_sources(source_records))

    report_coverage(sources=source_records, per_domain=PER_DOMAIN)

    section('Scenario drafts')
    drafts = build_drafts(sources=source_records)
    drafts, kept = merge_drafts(drafts=drafts, drafts_path=DRAFTS_PATH)
    drafts.to_csv(DRAFTS_PATH, index=False)
    print(f'Drafts: {shape_of(drafts)}')
    print(f'Kept {kept} written values')
    print(f'Drafts written: {int((drafts["request"].str.strip() != "").sum())} '
          f'of {len(drafts)}')
    report('drafts.csv', check_drafts(drafts))

    report_selection(drafts=drafts, domains=DOMAIN_NAMES, types=TYPES)

    section('Scenario slots')
    scenarios = build_scenarios(drafts=drafts, domains=DOMAIN_NAMES, types=TYPES)
    scenarios.to_csv(SCENARIOS_PATH, index=False)
    print(f'Scenarios: {shape_of(scenarios)}')
    print(f'Scenarios filled: {len(written(scenarios))} of {len(scenarios)}')
    report('scenarios.csv', check_scenarios(written(scenarios)))
