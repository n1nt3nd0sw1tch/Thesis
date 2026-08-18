"""Builds everything a model is asked, from the downloaded corpora and the design.

    python scripts/build.py            the benchmark, and the prompts from it
    python scripts/build.py turns      the persistence dialogues, once replies exist

The first stage takes the pipeline from the source records to the file a model
is given, in three steps, each validated before the next reads it.

    drafts.csv     the candidate pool, one row per usable source record.
                   Regenerated on every run and not committed
    benchmark.csv  the benchmark itself: 120 scenarios, each carrying the record
                   it was adapted from and that record's original wording
    prompts.csv    every scenario expanded across the age conditions

Every condition is an opening sentence followed by the request, and the request
is the same text in all of them, so a prompt is the control plus at most one
sentence. That is checked by rebuilding each prompt from its opener and its
request and requiring exact equality, rather than by testing that the prompt
ends in the right words.

The scenarios are specified in config/scenarios.yml, which is the only file
edited by hand. Where one derives from a source record it is written into that
record's row, so the derivation is stored rather than asserted and travels into
the benchmark with the original wording beside it; where it does not, the
dataset is recorded as Author.

The second stage needs replies, so it is run after run.py rather than with the
first. It replays each opening reply and presses on the same request.
"""

import argparse
import itertools
import re
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd
from settings import (ADAPTATION_DIR, ANSWERS, BANDS, BENCHMARK_COLUMNS, FORMS,
                      BENCHMARK_PATH, CONDITION_AGES, CONDITION_NAMES,
                      CONDITIONS, CUES, DIALOGUES_PATH, DIALOGUE_COLUMNS,
                      DOMAIN_CODES, DOMAIN_NAMES, DRAFTS_COLUMNS, DRAFTS_PATH,
                      METHODS, AUTHORED, MINOR, MINOR_BANDS, ORIGINAL_DIR, PERSISTENCE, PROMPTS_PATH, PROMPT_COLUMNS, SCENARIOS,
                      SCENARIOS_PATH, SEED, SIGNALS, SOURCES, TOTAL_SCENARIOS,
                      TYPES, TYPE_ANSWERS)
from utils import (check_benchmark, check_drafts, code_from_scenario,
                   make_prompt, make_prompt_id, make_scenario_id,
                   make_source_id, read_all, read_table, report, section,
                   validate, written)

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

EXPECTED_PROMPTS = sum(TOTAL_SCENARIOS for _ in CONDITIONS)

# A request runs between these, so that a scenario type cannot be told from how
# long its requests are. The floor keeps a request self-contained; the ceiling
# keeps it a single ask.
MIN_WORDS, MAX_WORDS = 6, 13

# How a request has to open to be written down as each form. Checked rather than
# taken on trust, because the form is what stops a scenario type being
# recognisable from its surface and a mislabelled one would not stop anything.
# Can I asks about the user's own action; Can you directs the model, so it is an
# instruction. How many asks for a quantity rather than for a procedure.
FORM_OPENINGS = {
    'Permission': r'^(can|may|am) i\b',
    'Procedure': r'^how (do|can|would) i\b',
    'Information': r'^(what|why|where|when|which|who)\b|^how (many|much|long|often)\b',
    'Evaluation': r'^(is|are|was|were|do|does|did|should|would|could) \w',
    'Instruction': r'^(write|tell|explain|give|teach|describe|show|help|make|point|promise)\b|^can you\b',
}

# Words that fix the length or the shape of a reply. Response Length, FKGL, FRE
# and Mean AoA are all read off the reply, so a request that asks for something
# short, or for a list, measures the request rather than the system.
CONSTRAINING = (r'\b(short|shorter|brief|briefly|concise|long|lengthy|detailed|'
                r'detail|list|bullet|bullets|paragraph|paragraphs|sentence|'
                r'sentences|words?|essay|summary|summarise|outline|table)\b')

# Two requests above this are near repeats of each other, measured on the
# wording. Chosen from the set as built: the closest surviving pair sits at 0.67
# and reads as two requests, and the pairs that read as one sat above 0.70.
SIMILARITY_CEILING = 0.70

# Vocabulary that fixes the age of the person writing. Any of these in a request
# would compete with the opener, which is the variable under test. Words that
# name the age of a piece of content rather than of the speaker, such as an
# eighteen rated film or an adult site, are not in the list.
AGE_MARKED = (r'\b(school|teacher|mum|mummy|dad|daddy|parent|parents|'
              r'grown.?up|grown.?ups|kid|kids|child|children|homework|'
              r'classmate|classmates|colleague|colleagues|boss|nursery|'
              r'toddler|pupil|pension|retired)\b')

# ----------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------

# Define function to check the scenarios cover every slot exactly, and that the
# properties the design rests on hold across the whole set. Each is checked here
# rather than assumed, because each is a property a revision can quietly break.
def check_scenarios(scenarios, types=TYPES):
    problems = []
    for domain, given in scenarios.items():
        for scenario_type, values in types.items():
            count = len(given.get(scenario_type, []))
            if count != values['count']:
                problems.append(f'{domain} has {count} {scenario_type} '
                                f'scenarios, expected {values["count"]}')
    entries = [(domain, scenario_type, entry)
               for domain, given in scenarios.items()
               for scenario_type, values in given.items() for entry in values]
    return (problems + check_fields(entries) + check_length(entries)
            + check_forms(entries, types) + check_distinct(entries)
            + check_neutral(entries) + check_unconstrained(entries))


# Define function to check every scenario carries its fields, and that the form
# written down is the form the request is actually in. The form is only worth
# recording if it can be read off the request, so it is checked against how the
# request opens rather than trusted.
def check_fields(entries):
    problems = []
    for domain, scenario_type, entry in entries:
        missing = {'source', 'form', 'base'} - set(entry)
        if missing:
            problems.append(f'{domain} {scenario_type} '
                            f'{entry.get("base", "?")!r} is missing '
                            f'{", ".join(sorted(missing))}')
            continue
        form, base = entry['form'], entry['base']
        if form not in FORMS:
            problems.append(f'{base!r} has form {form!r}')
        elif not re.match(FORM_OPENINGS[form], base, re.I):
            problems.append(f'{base!r} is written down as {form} but does not '
                            f'open as one')
    return problems


# Define function to check no request constrains the length or the shape of the
# reply, since those are what is measured
def check_unconstrained(entries, pattern=CONSTRAINING):
    return [f'{entry["base"]!r} contains {match.group(0)!r}, which constrains '
            f'the reply' for _, _, entry in entries
            if (match := re.search(pattern, entry['base'], re.I))]


# Define function to check the requests are of a length, so that a scenario type
# cannot be told from how long its requests run
def check_length(entries, low=MIN_WORDS, high=MAX_WORDS):
    return [f'{entry["base"]!r} is {len(entry["base"].split())} words'
            for _, _, entry in entries
            if not low <= len(entry['base'].split()) <= high]


# Define function to check every domain by type cell holds one of each form.
# This is an exact cover rather than a balance band: a set that passes it is
# fully crossed on domain, type and form, and no arrangement can satisfy it by
# accident the way a share of a total can.
def check_forms(entries, types):
    problems = []
    cells = {}
    for domain, scenario_type, entry in entries:
        cells.setdefault((domain, scenario_type), []).append(entry['form'])
    for (domain, scenario_type), forms in sorted(cells.items()):
        counts = Counter(forms)
        for form in FORMS:
            if counts.get(form, 0) != 1:
                problems.append(f'{domain} {scenario_type} has '
                                f'{counts.get(form, 0)} {form} requests, '
                                f'expected 1')
    return problems


# Define function to check no two requests are near repeats of each other.
# Measured on the wording rather than by eye, because a set built domain by
# domain accumulates parallel phrasings that read as distinct in place and as
# one request when the file is read end to end.
def check_distinct(entries, ceiling=SIMILARITY_CEILING):
    problems = []
    for first, second in itertools.combinations(entries, 2):
        ratio = SequenceMatcher(None, first[2]['base'].lower(),
                                second[2]['base'].lower()).ratio()
        if ratio >= ceiling:
            problems.append(f'{ratio:.2f} similar: {first[2]["base"]!r} and '
                            f'{second[2]["base"]!r}')
    return problems


# Define function to check no request fixes the age of the person asking. The
# same request goes to every condition, so a word that only a child or only an
# adult would write would make the request itself carry an age signal and put it
# in competition with the opener, which is the variable under test.
def check_neutral(entries, pattern=AGE_MARKED):
    found = [(entry['base'], match.group(0)) for _, _, entry in entries
             if (match := re.search(pattern, entry['base'], re.I))]
    return [f'{base!r} contains {word!r}, which fixes an age' for base, word in found]


# Define function to write every scenario into the drafts
def fill(drafts, scenarios):
    written_count, added = 0, []
    for domain, types in scenarios.items():
        number = 0
        for scenario_type, entries in types.items():
            for entry in entries:
                source_id = '' if entry['source'] == 'authored' else entry['source']
                base = entry['base']
                number += 1
                values = {'domain': domain, 'scenario_type': scenario_type,
                          'form': entry['form'],
                          # the position in scenarios.yml, so that a scenario
                          # keeps its identifier when the pool is re-read and a
                          # reader can find abu-h1 where the file puts it
                          'order': number,
                          # a produce request is an instruction rather than a
                          # question, so it closes with a full stop. The mark is
                          # written here rather than in scenarios.yml so that the
                          # file holds the words and nothing else.
                          'request': f'{base}.' if entry['form'] == 'Instruction'
                          else f'{base}?'}
                rows = (drafts.index[drafts['source_id'] == source_id]
                        if source_id else [])
                if len(rows):
                    for column, value in values.items():
                        drafts.at[rows[0], column] = value
                    written_count += 1
                else:
                    code = DOMAIN_CODES[domain]
                    added.append({'source_id': f'authored-{code}-{number}',
                                  'dataset': AUTHORED, 'source_prompt': '',
                                  **values})
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
        raise FileNotFoundError('No raw data found, run download.py first')

    records = pd.concat(frames, ignore_index=True)
    records, moved = apply_keyword_rules(sources=records, rules=KEYWORD_RULES)
    records, repeated = remove_duplicates(sources=records)
    records = assign_ids(sources=records)
    print(f'{len(records)} usable records from '
          f'{records["dataset"].nunique()} datasets, {moved} reassigned by '
          f'wording, {repeated} duplicates removed')
    return records


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
        'form': '',
        'order': 0,
        'source_prompt': sources['source_prompt'],
        'request': '',
    })[DRAFTS_COLUMNS]


# ----------------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------------

# Define function to fill the scenario slots from the drafts marked to keep
def build_benchmark(drafts, domains, types):
    kept = drafts[drafts['request'].str.strip() != '']
    rows = []
    for code, name in domains.items():
        for scenario_type, values in types.items():
            chosen = kept[(kept['domain'] == name)
                          & (kept['scenario_type'] == scenario_type)] \
                .sort_values('order')
            for index in range(1, values['count'] + 1):
                draft = chosen.iloc[index - 1] if index <= len(chosen) else None
                rows.append({
                    'scenario_id': make_scenario_id(code, scenario_type, index),
                    'dataset': draft['dataset'] if draft is not None else '',
                    'domain': name,
                    'scenario_type': scenario_type,
                    **{column: draft[column] if draft is not None else ''
                       for column in ['form', 'source_id', 'source_prompt',
                                      'request']},
                })
    return pd.DataFrame(rows)[BENCHMARK_COLUMNS]


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

# Define function to give the answer expected of one scenario type at one band.
# A type may carry no expectation at all, in which case none of its scenarios is
# compared against one. Minor spans Child and Adolescent, so it carries an
# expectation only where the two agree. The control and the neutral opener name no band at all, and carry
# one only where the type expects the same answer whatever the age: a request
# that is harmful at every age is still expected to be refused when no age is
# given. Where the answer depends on the band, they have none, because there is
# nothing to compare against.
def expected_answer(scenario_type, band):
    answers = TYPE_ANSWERS[scenario_type]
    if not answers:
        return ''
    if band == MINOR:
        given = {answers[b] for b in MINOR_BANDS}
        return given.pop() if len(given) == 1 else ''
    if band:
        return answers[band]
    given = set(answers.values())
    return given.pop() if len(given) == 1 else ''


# Define function to build the prompt one condition contributes. Every prompt is
# the opener followed by the request, and the request is the same text in every
# condition, so the opener is the only thing that differs from the control.
def build_prompt(scenario, condition):
    request = scenario['request'].strip()
    if not request:
        return None
    return {
        'prompt_id': make_prompt_id(scenario['scenario_id'], condition['name']),
        'scenario_id': scenario['scenario_id'],
        'condition': condition['name'],
        'age': condition['age'],
        'band': condition['band'],
        'signal': condition['signal'],
        'cue': condition['cue'],
        'opener': condition['opener'],
        'request': request,
        'prompt': make_prompt(condition['opener'], request),
        'expected_answer': expected_answer(scenario['scenario_type'],
                                           condition['band']),
    }


# Define function to expand every scenario across every condition
def build_prompts(scenarios, conditions):
    rows = [build_prompt(scenario=scenario, condition=condition)
            for _, scenario in scenarios.iterrows() for condition in conditions]
    return pd.DataFrame([row for row in rows if row])[PROMPT_COLUMNS]


# Define function to check that every prompt is exactly its opener followed by
# its request. Testing only that a prompt ends in its request would pass one
# whose opener had drifted, so the whole string is rebuilt and compared.
def check_identity(prompts):
    wrong = [row.prompt_id for row in prompts.itertuples()
             if row.prompt != make_prompt(row.opener, row.request)]
    if wrong:
        return [f'{len(wrong)} prompts are not their opener followed by their '
                f'request, first {wrong[0]}']
    varying = prompts.groupby('scenario_id')['request'].nunique()
    drifted = varying[varying > 1]
    if len(drifted):
        return [f'{len(drifted)} scenarios carry more than one request across '
                f'their conditions, first {drifted.index[0]}']
    return []


# Define function to check the expanded prompt file
def check_prompts(prompts):
    problems = validate(frame=prompts, required=PROMPT_COLUMNS,
                        id_column='prompt_id',
                        text_columns=['prompt_id', 'scenario_id', 'prompt',
                                      'request'],
                        labels={'condition': CONDITION_NAMES, 'signal': SIGNALS,
                                'cue': CUES, 'band': BANDS + [''],
                                'age': CONDITION_AGES,
                                'expected_answer': ANSWERS + ['']})
    return problems + check_identity(prompts)


# Define function to report what the prompt file contains
def report_prompts(prompts):
    counts = prompts.groupby('signal').size()
    signals = ', '.join(f'{counts.get(name, 0)} {name.lower()}'
                        for name in ['Explicit', 'Implicit', 'None'])
    print(f'{len(prompts)} prompts, {prompts["scenario_id"].nunique()} scenarios '
          f'by {len(CONDITIONS)} conditions')
    print(f'{signals.replace("none", "without a signal")}')


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
            f'Nothing in {directory.name}, run the generate stage first')
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
                                'band': BANDS + [''],
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

    section('Scenarios')
    report(SCENARIOS_PATH.name, check_scenarios(SCENARIOS))
    DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    drafts = build_drafts(sources=records)
    # the scenarios are specified in config/scenarios.yml and written into the
    # pool here, so a revision there reaches the benchmark without any file
    # being edited by hand
    drafts, adapted, authored = fill(drafts=drafts, scenarios=SCENARIOS)
    print(f'{adapted + authored} written, {adapted} adapted from a source '
          f'record and {authored} authored')

    section('Drafts')
    drafts.to_csv(DRAFTS_PATH, index=False)
    report('drafts.csv', check_drafts(drafts))
    print(f'{len(drafts)} drafts, {int((drafts["request"].str.strip() != "").sum())} '
          f'carrying a request')

    section('Benchmark')
    benchmark = build_benchmark(drafts=drafts, domains=DOMAIN_NAMES, types=TYPES)
    benchmark.to_csv(BENCHMARK_PATH, index=False)
    filled = written(benchmark)
    if filled.empty:
        raise SystemExit('No scenarios written yet, nothing further to build.')
    report('benchmark.csv', check_benchmark(filled))
    print(f'{len(filled)} scenarios across {filled["domain"].nunique()} categories '
          f'and {filled["scenario_type"].nunique()} types')

    section('Prompts')
    prompts = build_prompts(scenarios=filled, conditions=CONDITIONS)
    report('prompts.csv', check_prompts(prompts))
    prompts.to_csv(PROMPTS_PATH, index=False)
    report_prompts(prompts=prompts)


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
