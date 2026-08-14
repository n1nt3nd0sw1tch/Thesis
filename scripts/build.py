"""Builds everything a model is asked, from the downloaded corpora and the design.

    python scripts/build.py            the benchmark, and the prompts from it
    python scripts/build.py turns      the persistence dialogues, once replies exist

The first stage takes the pipeline from the source records to the file a model
is given, in four steps, each validated before the next reads it.

    drafts.csv     one draft per usable source record, and the only file edited
                   by hand. Work already in it is preserved on every run
    benchmark.csv  the 120 scenario slots, filled from the drafts marked to keep
    prompts.csv    every scenario expanded across the eleven age conditions
    scores.csv     the readability of each request variant, which is the check
                   that a variant differs from the canonical request in its cue

The scenarios themselves are specified in config/scenarios.yml as a base and a
cue clause. The base is the canonical request and the clause is appended to make
each variant, so a variant is the control plus one phrase and the four texts of
a scenario differ in exactly one contiguous span. Building them here rather than
editing four columns by hand is what keeps that property true after a revision.
Where a scenario derives from a source record it is written into that record's
row, so the derivation is stored rather than asserted; where it does not, a row
is added with the dataset left blank.

The second stage needs replies, so it is run after run.py rather than with the
first. It replays each opening reply and presses on the same request.
"""

import argparse

import pandas as pd
from evaluate import load_aoa, score_frame
from scripts.settings import (ADAPTATION_DIR, AGE_BANDS, ANSWERS, BENCHMARK_COLUMNS,
                      BENCHMARK_PATH, CONDITION_AGES, CONDITION_NAMES,
                      CONDITIONS, CUES, DIALOGUES_PATH, DIALOGUE_COLUMNS,
                      DOMAIN_CODES, DOMAIN_NAMES, DRAFTS_COLUMNS, DRAFTS_PATH,
                      METHODS, ORIGINAL_DIR, PER_DOMAIN, PERSISTENCE,
                      PROMPTS_PATH, PROMPT_COLUMNS, SCENARIOS, SCENARIOS_PATH,
                      SCORES_PATH, SCORE_COLUMNS, SEED, SIGNALS, SOURCES,
                      TOTAL_SCENARIOS, TYPES, TYPE_ANSWERS, VARIANT_COLUMNS,
                      WHO, variant_column)
from utils import (check_benchmark, check_drafts, code_from_scenario,
                   make_prompt, make_prompt_id, make_scenario_id,
                   make_source_id, read_all, read_table, report, section,
                   shape_of, validate, written)

# ----------------------------------------------------------------------------
# Settings
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

# The draft columns written by hand. All others are generated on every run.
DRAFT_FIELDS = ['scenario_type', *VARIANT_COLUMNS, 'keep']

EXPECTED_PROMPTS = TOTAL_SCENARIOS * len(CONDITIONS)

# ----------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------

# Define function to build the four texts of one scenario from its base
def build_texts(base, clause):
    return {'request': f'{base}?',
            **{column: f'{base} {clause.format(who=who)}?'
               for column, who in WHO.items()}}


# Define function to check the scenarios cover every slot exactly
def check_scenarios(scenarios, types=TYPES):
    problems = []
    for domain, given in scenarios.items():
        for scenario_type, values in types.items():
            count = len(given.get(scenario_type, []))
            if count != values['count']:
                problems.append(f'{domain} has {count} {scenario_type} '
                                f'scenarios, expected {values["count"]}')
    bases = [entry['base'] for types_ in scenarios.values()
             for entries in types_.values() for entry in entries]
    repeated = {base for base in bases if bases.count(base) > 1}
    if repeated:
        problems.append(f'{len(repeated)} bases appear more than once')
    return problems


# Define function to write every scenario into the drafts
def fill(drafts, scenarios):
    # rows added on a previous run are dropped first, so running twice gives the
    # same file as running once and a removed scenario leaves nothing behind
    drafts = drafts[~drafts['source_id'].str.startswith('authored-')].copy()
    written_count, added = 0, []
    for domain, types in scenarios.items():
        number = 0
        for scenario_type, entries in types.items():
            for entry in entries:
                source_id = '' if entry['source'] == 'authored' else entry['source']
                base, clause = entry['base'], entry['cue']
                number += 1
                values = {'domain': domain, 'scenario_type': scenario_type,
                          'implicit_cue': 'People', 'keep': 'yes',
                          **build_texts(base, clause)}
                rows = (drafts.index[drafts['source_id'] == source_id]
                        if source_id else [])
                if len(rows):
                    for column, value in values.items():
                        drafts.at[rows[0], column] = value
                    written_count += 1
                else:
                    code = DOMAIN_CODES[domain]
                    added.append({'source_id': f'authored-{code}-{number}',
                                  'dataset': '', 'source_prompt': '', **values})
    filled = pd.concat([drafts, pd.DataFrame(added)], ignore_index=True)
    return filled[DRAFTS_COLUMNS], written_count, len(added)


# ----------------------------------------------------------------------------
# Source records
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


# Define function to collect the source records of every dataset
def build_sources(sources, original_dir):
    frames = [select(name=name, spec=spec, original_dir=original_dir)
              for name, spec in sources.items()]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise FileNotFoundError('no raw data found, run download.py first')

    records = pd.concat(frames, ignore_index=True)
    records, moved = apply_keyword_rules(sources=records, rules=KEYWORD_RULES)
    records, repeated = remove_duplicates(sources=records)
    records = assign_ids(sources=records)
    print(f'{len(records)} usable records from '
          f'{records["dataset"].nunique()} datasets, {moved} reassigned by '
          f'wording, {repeated} duplicates removed')
    return records


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


# ----------------------------------------------------------------------------
# Drafts
# ----------------------------------------------------------------------------

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
    previous = read_table(drafts_path)
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


# ----------------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

# Define function to pick the request a condition asks for
def request_for(scenario, condition):
    column = (variant_column(condition['variant']) if condition['variant']
              else 'request')
    return scenario[column].strip()


# Define function to build the prompt one condition contributes, if it has one
def build_prompt(scenario, condition):
    request = request_for(scenario, condition)
    if not request:
        return None
    band = condition['band']
    return {
        'prompt_id': make_prompt_id(scenario['scenario_id'], condition['name']),
        'scenario_id': scenario['scenario_id'],
        'condition': condition['name'],
        'age': condition['age'],
        'band': band,
        'signal': condition['signal'],
        'cue': condition['cue'] or scenario['implicit_cue'],
        'prompt': make_prompt(condition['opener'], request),
        'expected_answer': TYPE_ANSWERS[scenario['scenario_type']][
            AGE_BANDS.index(band)] if band else '',
    }


# Define function to expand every scenario across every condition
def build_prompts(scenarios, conditions):
    rows = [build_prompt(scenario=scenario, condition=condition)
            for _, scenario in scenarios.iterrows() for condition in conditions]
    return pd.DataFrame([row for row in rows if row])[PROMPT_COLUMNS]


# Define function to check that every prompt ends in the request it asked for
def check_matching(prompts, scenarios):
    by_condition = {condition['name']: condition for condition in CONDITIONS}
    indexed = scenarios.set_index('scenario_id')
    unmatched = [row.prompt_id for row in prompts.itertuples()
                 if not row.prompt.endswith(
                     request_for(indexed.loc[row.scenario_id],
                                 by_condition[row.condition]))]
    if unmatched:
        return [f'{len(unmatched)} prompts do not end in the request their '
                f'condition asked for, first {unmatched[0]}']
    return []


# Define function to check the expanded prompt file
def check_prompts(prompts, scenarios):
    problems = validate(frame=prompts, required=PROMPT_COLUMNS,
                        id_column='prompt_id',
                        text_columns=['prompt_id', 'scenario_id', 'prompt'],
                        labels={'condition': CONDITION_NAMES, 'signal': SIGNALS,
                                'cue': CUES, 'band': AGE_BANDS + [''],
                                'age': CONDITION_AGES,
                                'expected_answer': ANSWERS + ['']})
    return problems + check_matching(prompts=prompts, scenarios=scenarios)


# Define function to report what the prompt file contains
def report_prompts(prompts):
    signals = prompts.groupby('signal').size()
    print(f'{len(prompts)} prompts, ' + ', '.join(
        f'{count} {signal.lower()}' for signal, count in signals.items()))
    if len(prompts) != EXPECTED_PROMPTS:
        print(f'Expected: {EXPECTED_PROMPTS} prompts once every scenario is written')


# ----------------------------------------------------------------------------
# Request scores
# ----------------------------------------------------------------------------

# Define function to score the canonical request and every cue variant
def score_variants(benchmark, norms):
    frames = [score_frame(benchmark, 'canonical', norms)]
    for band in AGE_BANDS:
        column = variant_column(band)
        variants = benchmark[benchmark[column].str.strip() != '']
        if not variants.empty:
            frames.append(score_frame(variants, column, norms, column=column))
    return pd.concat(frames, ignore_index=True)[SCORE_COLUMNS]


# Define function to compare the variants on every measure
def report_comparison(scores):
    measures = ['words', 'fkgl', 'fre', 'mean_aoa', 'max_aoa', 'difficult',
                'covered']
    section('Mean score per variant')
    print(scores.groupby('variant')[measures].mean().round(2).to_string())


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# The persistence dialogues
# ----------------------------------------------------------------------------

# Each dialogue opens with a prompt already put to a system and that system's
# own reply to it, then presses on the same request. Replaying an observed reply
# rather than generating a fresh one holds the starting point constant, so later
# behaviour is measured against what the system actually did. The wording of each
# method is identical across scenarios, conditions and systems, so only depth and
# method differ from the single-turn case. Only the assistant turns after the
# first are generated.

# Columns expected of the single-turn replies collected beforehand
RESPONSE_COLUMNS = ['prompt_id', 'model', 'replicate', 'response', 'error']

# Define function to read the single-turn replies the dialogues open with
def load_responses(directory=ADAPTATION_DIR):
    responses = read_all(directory)
    if responses.empty:
        raise FileNotFoundError(
            f'nothing in {directory.name}, run the generate stage first')
    missing = [column for column in RESPONSE_COLUMNS
               if column not in responses.columns]
    if missing:
        raise KeyError(f'{directory.name} is missing columns '
                       f'{", ".join(missing)}')
    responses = responses[responses['error'].astype(str).str.strip() == '']
    return responses.astype({'replicate': str})


# Define function to choose the scenarios the extension runs on
def choose_scenarios(prompts, count, seed):
    scenarios = prompts[['scenario_id']].drop_duplicates()
    scenarios['domain'] = scenarios['scenario_id'].map(code_from_scenario)
    scenarios['scenario_type'] = scenarios['scenario_id'].map(
        lambda name: next(kind for kind, values in TYPES.items()
                          if values['code'] == name.split('-')[1][0]))
    chosen = (scenarios.groupby(['domain', 'scenario_type'], group_keys=False)
              .apply(lambda group: group.sample(
                  n=min(len(group), max(1, round(count / len(scenarios)
                                                 * len(group)))),
                  random_state=seed)))
    return sorted(chosen['scenario_id'])


# Define function to build one dialogue from a prompt, its reply, and a method
def build_dialogue(prompt, reply, method, turns, request):
    slug = method.lower().replace(' ', '-')
    dialogue_id = (f'{prompt["prompt_id"]}-{reply["model"]}'
                   f'-r{reply["replicate"]}-{slug}')
    shared = {'dialogue_id': dialogue_id, 'prompt_id': prompt['prompt_id'],
              'scenario_id': prompt['scenario_id'],
              'condition': prompt['condition'], 'band': prompt['band'],
              'model': reply['model'], 'opening_replicate': reply['replicate'],
              'method': method}
    rows = [{**shared, 'turn': 1, 'role': 'user', 'text': prompt['prompt'],
             'expected_answer': prompt['expected_answer']},
            {**shared, 'turn': 2, 'role': 'assistant', 'text': reply['response'],
             'expected_answer': ''}]
    for index, wording in enumerate(turns):
        turn = 3 + index * 2
        rows.append({**shared, 'turn': turn, 'role': 'user',
                     'text': wording.format(request=request),
                     'expected_answer': ''})
        rows.append({**shared, 'turn': turn + 1, 'role': 'assistant', 'text': '',
                     'expected_answer': prompt['expected_answer']})
    return rows


# Define function to build every dialogue the extension needs
def build_dialogues(prompts, responses, requests, methods, scenarios,
                    conditions, opening_replicate):
    wanted = prompts[prompts['scenario_id'].isin(scenarios)
                     & prompts['condition'].isin(conditions)]
    opening = responses[responses['replicate'] == str(opening_replicate)]
    merged = wanted.merge(opening, on='prompt_id', how='inner')
    rows = [row for _, pair in merged.iterrows() for method in methods
            for row in build_dialogue(prompt=pair, reply=pair, method=method,
                                      turns=METHODS[method]['turns'],
                                      request=requests[pair['scenario_id']])]
    return pd.DataFrame(rows)[DIALOGUE_COLUMNS]


# Define function to check the dialogue file
def check_dialogues(dialogues, methods):
    problems = validate(frame=dialogues, required=DIALOGUE_COLUMNS,
                        text_columns=['dialogue_id', 'prompt_id', 'scenario_id'],
                        labels={'role': ['user', 'assistant'],
                                'band': AGE_BANDS + [''],
                                'method': list(METHODS),
                                'expected_answer': ANSWERS + ['']})
    turns = 2 + 2 * len(METHODS[methods[0]]['turns'])
    counts = dialogues.groupby('dialogue_id').size()
    uneven = counts[counts != turns]
    if len(uneven):
        problems.append(f'{len(uneven)} dialogues do not have {turns} turns')

    # turn arrives as text when the file is read back, so compare numerically
    numbered = pd.to_numeric(dialogues['turn'], errors='coerce')
    replayed = dialogues[(numbered == 2)
                         & (dialogues['text'].astype(str).str.strip() == '')]
    if len(replayed):
        problems.append(f'{len(replayed)} dialogues have an empty replayed reply')

    generated = dialogues[(numbered > 2) & (dialogues['role'] == 'assistant')
                          & (dialogues['text'].astype(str).str.strip() != '')]
    if len(generated):
        problems.append(f'{len(generated)} later assistant turns are already '
                        f'filled, which should happen at generation time')
    return problems


# Define function to report what the dialogue file contains
def report_dialogues(dialogues, methods, scenarios):
    numbered = pd.to_numeric(dialogues['turn'], errors='coerce')
    generated = int(((dialogues['role'] == 'assistant') & (numbered > 2)).sum())
    print(f'{dialogues["dialogue_id"].nunique()} conversations from '
          f'{len(scenarios)} scenarios, '
          f'{2 + 2 * len(METHODS[methods[0]]["turns"])} turns each, '
          f'{generated} replies to generate')
    opening = dialogues[numbered == 1]
    print(pd.crosstab(opening['condition'], opening['method'],
                      margins=True, margins_name='total').to_string())

# Define function to build the benchmark, the prompts and the request scores
def build_all():
    section('Source records')
    records = build_sources(sources=SOURCES, original_dir=ORIGINAL_DIR)
    report_coverage(sources=records, per_domain=PER_DOMAIN)

    section('Scenarios')
    report(SCENARIOS_PATH.name, check_scenarios(SCENARIOS))

    DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    drafts = build_drafts(sources=records)
    drafts, _ = merge_drafts(drafts=drafts, drafts_path=DRAFTS_PATH)
    # the scenarios are specified in config/scenarios.yml and written into the
    # pool here, so a revision there reaches the benchmark without any file
    # being edited by hand
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
    filled = written(benchmark)
    print(f'{len(benchmark)} scenarios, {len(filled)} filled')
    report('benchmark.csv', check_benchmark(filled))
    if filled.empty:
        raise SystemExit('No scenarios written yet, nothing further to build.')

    section('Model prompts')
    prompts = build_prompts(scenarios=filled, conditions=CONDITIONS)
    report('prompts.csv', check_prompts(prompts=prompts, scenarios=filled))
    prompts.to_csv(PROMPTS_PATH, index=False)
    report_prompts(prompts=prompts)

    section('Request scores')
    scores = score_variants(benchmark=filled, norms=load_aoa())
    print(f'Scores: {shape_of(scores)}')
    report('scores.csv', validate(frame=scores, required=SCORE_COLUMNS,
                                  text_columns=['variant', 'scenario_id']))
    scores.to_csv(SCORES_PATH, index=False)
    report_comparison(scores=scores)


# Define function to build the replayed dialogues of the persistence extension
def build_turns():
    section('Persistence extension')
    prompts = read_table(PROMPTS_PATH)
    responses = load_responses()
    benchmark = read_table(BENCHMARK_PATH)
    # topic change returns to the request alone, without the opening sentence,
    # so that the age is not restated at the turn being scored
    requests = dict(zip(benchmark['scenario_id'], benchmark['request']))

    scenarios = choose_scenarios(prompts=prompts,
                                 count=PERSISTENCE['scenarios'], seed=SEED)
    methods = PERSISTENCE['methods']
    dialogues = build_dialogues(prompts=prompts, responses=responses,
                                requests=requests, methods=methods,
                                scenarios=scenarios,
                                conditions=PERSISTENCE['conditions'],
                                opening_replicate=PERSISTENCE['opening_replicate'])
    report('dialogues.csv', check_dialogues(dialogues=dialogues, methods=methods))

    DIALOGUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    dialogues.to_csv(DIALOGUES_PATH, index=False)
    report_dialogues(dialogues=dialogues, methods=methods, scenarios=scenarios)


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', nargs='?', default='benchmark',
                        choices=['benchmark', 'turns'])
    arguments = parser.parse_args()
    build_all() if arguments.stage == 'benchmark' else build_turns()
