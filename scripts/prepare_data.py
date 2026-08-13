"""Turns the downloaded datasets into two files. drafts.csv opens one draft per
usable source record and is the only file written by hand; benchmark.csv is the
120 slots, filled from the drafts marked to keep. Work in drafts.csv is preserved
on every run."""

import pandas as pd
from fill_scenarios import check_scenarios, fill
from settings import (BENCHMARK_COLUMNS, BENCHMARK_PATH, SCENARIOS, SCENARIOS_PATH,
                      DOMAIN_NAMES, DRAFTS_COLUMNS, DRAFTS_PATH, ORIGINAL_DIR,
                      PER_DOMAIN, SOURCES, TOTAL_SCENARIOS, TYPE_ANSWERS, TYPES,
                      check_benchmark, check_drafts, make_scenario_id,
                      VARIANT_COLUMNS, make_source_id, report, section,
                      shape_of, written)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Settings: reassignment and slot filling
# ----------------------------------------------------------------------------

# Body image and eating disorder material sits across several source label
# schemes without a label of its own, so it is identified by wording and
# reassigned after the label mappings. The patterns do not overlap, body image
# is checked first, and the word stems match their inflected forms.
KEYWORD_RULES = {
    'bod': (r'\b(?:body image|look fat|too fat|fat people|skinny girls|'
            r'how much should i weigh|ugly)\b'),
    'eat': (r'\b(?:anorex|bulimi|purg|starv|throw up|vomit|thinspo|fasting|'
            r'calorie restriction)\w*'),
    'vio': r'\b(?:assassinat\w*|john f kennedy|shinzo abe)',
}

# The draft columns written by hand, all others are generated on every run.
DRAFT_FIELDS = ['scenario_type', *VARIANT_COLUMNS, 'keep']

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


# Define function to read a labelled column, cut back to its leading phrase
def read_labels(frame, label, split=''):
    labels = frame[label].astype(str)
    return labels.str.split(split).str[0].str.strip() if split else labels.str.strip()


# Define function to read a domain code out of one labelled column
def map_domains(frame, label, domains, split=''):
    lowered = {str(value).lower(): code for code, values in domains.items()
               for value in values}
    return read_labels(frame, label, split).str.lower().map(lowered)


# Define function to read the source records of one dataset into a common shape
def select(name, spec, original_dir):
    frame = load_original(filename=spec['file'], original_dir=original_dir)
    if frame is None:
        return None
    for column, allowed in spec.get('keep', {}).items():
        frame = frame[frame[column].isin(allowed)]
    if spec.get('exclude'):
        frame = frame[~frame[spec['record']].isin(spec['exclude'])]

    needed = [spec['text'], spec['label'], *spec.get('keep', {}),
              *([spec['record']] if spec['record'] else []),
              *([spec['fallback']['label']] if spec.get('fallback') else [])]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise KeyError(f'{spec["file"]} is missing columns {", ".join(missing)}')

    codes = map_domains(frame=frame, label=spec['label'], domains=spec['domains'],
                        split=spec.get('split', ''))
    if spec.get('fallback'):
        codes = codes.fillna(map_domains(frame=frame, **spec['fallback']))

    records = frame[spec['record']] if spec['record'] else frame.index
    selected = pd.DataFrame({
        'dataset': name,
        'record_id': list(records),
        'source_prompt': frame[spec['text']].astype(str).str.strip().tolist(),
        'domain_code': codes.tolist(),
    })
    return selected[selected['domain_code'].notna()].reset_index(drop=True)


# Define function to reassign records whose wording names a domain directly
def apply_keyword_rules(sources, rules):
    assigned = pd.Series(False, index=sources.index)
    for code, pattern in rules.items():
        matches = sources['source_prompt'].str.contains(
            pattern, case=False, regex=True) & ~assigned
        sources.loc[matches, 'domain_code'] = code
        assigned = assigned | matches
    return sources, int(assigned.sum())


# Define function to drop records repeating the wording of an earlier one
def remove_duplicates(sources):
    normalised = (sources['source_prompt'].str.lower()
                  .str.replace(r'[^a-z0-9\s]', '', regex=True)
                  .str.replace(r'\s+', ' ', regex=True).str.strip())
    repeated = sources.assign(normalised=normalised) \
        .duplicated(subset=['domain_code', 'normalised'])
    return sources.loc[~repeated].reset_index(drop=True), int(repeated.sum())


# Define function to give every source record its identifier and domain name
def assign_ids(sources):
    sources = sources.assign(
        source_id=[make_source_id(dataset, record) for dataset, record
                   in zip(sources['dataset'], sources['record_id'])],
        domain=sources['domain_code'].map(DOMAIN_NAMES))
    return sources.sort_values(['domain', 'source_id']).reset_index(drop=True)


# Define function to report how many source records each domain has
def report_coverage(sources, per_domain):
    counts = sources['domain_code'].value_counts()
    coverage = pd.DataFrame({
        'domain': list(DOMAIN_NAMES.values()),
        'available': [int(counts.get(code, 0)) for code in DOMAIN_NAMES],
    })
    coverage['to_author'] = (per_domain - coverage['available']).clip(lower=0)
    short = coverage[coverage['to_author'] > 0]
    if not short.empty:
        print(f'{int(short["to_author"].sum())} scenarios to write without a '
              f'source record:')
        print(short.to_string(index=False))


# Define function to open a draft for every source record
def build_drafts(sources):
    return pd.DataFrame({
        'source_id': sources['source_id'],
        'dataset': sources['dataset'].map(
            {name: spec['name'] for name, spec in SOURCES.items()}),
        'domain': sources['domain'],
        'scenario_type': sources['dataset'].map(
            {name: spec['scenario_type'] for name, spec in SOURCES.items()}),
        'source_prompt': sources['source_prompt'],
        **{column: '' for column in VARIANT_COLUMNS},
        'keep': '',
    })[DRAFTS_COLUMNS]


# Define function to carry over the drafts already written and marked
def merge_drafts(drafts, drafts_path):
    if not drafts_path.exists():
        return drafts, 0
    previous = pd.read_csv(drafts_path, dtype=str, keep_default_na=False).fillna('')
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
def build_benchmark(drafts, domains, types):
    kept = drafts[drafts['keep'].str.strip().str.lower() == 'yes']
    rows = []
    for code, name in domains.items():
        for scenario_type, values in types.items():
            chosen = kept[(kept['domain'] == name)
                          & (kept['scenario_type'] == scenario_type)]
            for index in range(1, values['count'] + 1):
                draft = chosen.iloc[index - 1] if index <= len(chosen) else None
                rows.append({
                    'scenario_id': make_scenario_id(code, scenario_type, index),
                    'dataset': draft['dataset'] if draft is not None else '',
                    'domain': name,
                    'scenario_type': scenario_type,
                    **{column: draft[column] if draft is not None else ''
                       for column in VARIANT_COLUMNS},
                })
    return pd.DataFrame(rows)[BENCHMARK_COLUMNS]


# Define function to report where too few or too many drafts have been kept
def report_selection(drafts, domains, types):
    kept = drafts[drafts['keep'].str.strip().str.lower() == 'yes']
    counts = kept.groupby(['domain', 'scenario_type']).size()
    selection = pd.DataFrame(
        [[int(counts.get((name, scenario_type), 0)) - values['count']
          for scenario_type, values in types.items()] for name in domains.values()],
        index=list(domains.values()), columns=list(types)).rename_axis('domain')
    section('Slots')
    print(selection.to_string())
    short = int(selection.clip(upper=0).abs().to_numpy().sum())
    spare = int(selection.clip(lower=0).to_numpy().sum())
    print(f'{short} slots short, {spare} kept drafts unused')


# Define function to report how the cue families are spread across the drafts kept
def report_cues(drafts):
    kept = drafts[(drafts['keep'].str.strip().str.lower() == 'yes')
                  & (drafts['implicit_cue'].str.strip() != '')]
    if kept.empty:
        return
    section('Cue families across the drafts kept')
    print(pd.crosstab(kept['domain'], kept['implicit_cue'],
                      margins=True, margins_name='total').to_string())



# Define function to collect the source records of every dataset
def build_sources(sources, original_dir):
    frames = [select(name=name, spec=spec, original_dir=original_dir)
              for name, spec in sources.items()]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise FileNotFoundError('no raw data found, run download_data.py first')

    records = pd.concat(frames, ignore_index=True)
    records, moved = apply_keyword_rules(sources=records, rules=KEYWORD_RULES)
    records, repeated = remove_duplicates(sources=records)
    records = assign_ids(sources=records)
    print(f'{len(records)} usable records from '
          f'{records["dataset"].nunique()} datasets, {moved} reassigned by '
          f'wording, {repeated} duplicates removed')
    return records


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    section('Source records')
    records = build_sources(sources=SOURCES, original_dir=ORIGINAL_DIR)
    report_coverage(sources=records, per_domain=PER_DOMAIN)

    section('Scenarios')
    report(SCENARIOS_PATH.name, check_scenarios(SCENARIOS))

    DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    drafts = build_drafts(sources=records)
    drafts, kept = merge_drafts(drafts=drafts, drafts_path=DRAFTS_PATH)
    # the scenarios are specified in scenarios.py and written into the pool here,
    # so a revision there reaches the benchmark without any file being edited
    drafts, sourced, authored = fill(drafts=drafts, scenarios=SCENARIOS)
    print(f'{sourced + authored} scenarios written, {sourced} into a source '
          f'record and {authored} without one')

    section('Scenario drafts')
    drafts.to_csv(DRAFTS_PATH, index=False)
    written_count = int((drafts['request'].str.strip() != '').sum())
    marked = int(drafts['keep'].str.strip().str.lower().eq('yes').sum())
    print(f'{len(drafts)} drafts, {written_count} requests written, '
          f'{marked} kept')
    report('drafts.csv', check_drafts(drafts))

    report_selection(drafts=drafts, domains=DOMAIN_NAMES, types=TYPES)
    report_cues(drafts=drafts)

    section('Benchmark')
    benchmark = build_benchmark(drafts=drafts, domains=DOMAIN_NAMES, types=TYPES)
    benchmark.to_csv(BENCHMARK_PATH, index=False)
    print(f'{len(benchmark)} scenarios, {len(written(benchmark))} filled')
    report('benchmark.csv', check_benchmark(written(benchmark)))
