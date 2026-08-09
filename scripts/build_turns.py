"""Turns the single-turn prompts into replayed dialogues for the persistence
extension. Each dialogue opens with a prompt already put to a system and that
system's own reply to it, then presses on the same request. Replaying an observed
reply rather than generating a fresh one holds the starting point constant, so
later behaviour is measured against what the system actually did. Three methods
press in different ways: probing insists, topic change steps away and returns, and
purpose reverse re-asks under a protective frame. The wording of each is identical
across scenarios, conditions and systems, so only depth and method differ from the
single-turn case. Only the assistant turns after the first are generated."""

import pandas as pd
from settings import (AGE_BANDS, BENCHMARK_PATH, DIALOGUE_COLUMNS,
                      DIALOGUES_PATH, ANSWERS, METHODS, PERSISTENCE,
                      PROMPTS_PATH, RESPONSES_PATH, SEED, TYPES,
                      code_from_scenario, report, section, shape_of, validate)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

# Columns expected of the single-turn responses collected beforehand
RESPONSE_COLUMNS = ['prompt_id', 'model', 'replicate', 'response']

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to read the single-turn responses the dialogues open with
def load_responses(responses_path):
    if not responses_path.exists():
        raise FileNotFoundError(
            f'{responses_path.name} not found, run the single-turn generation first')
    responses = pd.read_csv(responses_path, dtype=str, keep_default_na=False).fillna('')
    missing = [column for column in RESPONSE_COLUMNS
               if column not in responses.columns]
    if missing:
        raise KeyError(f'{responses_path.name} is missing columns '
                       f'{", ".join(missing)}')
    return responses


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

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    section('Persistence extension')
    prompts = pd.read_csv(PROMPTS_PATH, dtype=str, keep_default_na=False).fillna('')
    responses = load_responses(RESPONSES_PATH)
    benchmark = pd.read_csv(BENCHMARK_PATH, dtype=str, keep_default_na=False).fillna('')
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

    dialogues.to_csv(DIALOGUES_PATH, index=False)
    report_dialogues(dialogues=dialogues, methods=methods, scenarios=scenarios)
