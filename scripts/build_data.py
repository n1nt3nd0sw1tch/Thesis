"""Expands every written scenario across the age conditions into the model-ready
prompt file. Each prompt is a single turn: the condition opener carries the age
signal and is followed by the scenario request. The request is identical
across every condition, which is checked before the file is written."""

import pandas as pd
from settings import (ACTIONS, AGE_BANDS, CONDITION_AGES,
                      CONDITION_NAMES,
                      CONDITIONS, CUES, PROMPT_COLUMNS, PROMPTS_PATH,
                      BENCHMARK_PATH, SIGNALS, TOTAL_SCENARIOS, check_benchmark,
                      TYPE_ACTIONS, make_prompt, make_prompt_id, report,
                      section, shape_of, validate, written)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

EXPECTED_PROMPTS = TOTAL_SCENARIOS * len(CONDITIONS)

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to read the scenario worksheet and keep only written rows
def load_benchmark(benchmark_path):
    if not benchmark_path.exists():
        raise FileNotFoundError('benchmark.csv not found, run prepare_data.py first')
    scenarios = pd.read_csv(benchmark_path, dtype=str).fillna('')
    rows = written(scenarios)
    print(f'Scenarios written: {len(rows)} of {len(scenarios)}')
    return rows


# Define function to pick the prompt variant a condition asks for
def prompt_for(scenario, condition):
    column = f'implicit_{condition["request"]}' if condition['request'] else 'prompt'
    return scenario[column].strip()


# Define function to build the prompt one condition contributes, if it has one
def build_prompt(scenario, condition):
    variant = prompt_for(scenario, condition)
    if not variant:
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
        'prompt': make_prompt(condition['opener'], variant),
        'expected_action': TYPE_ACTIONS[scenario['scenario_type']][
            AGE_BANDS.index(band)] if band else '',
    }


# Define function to expand every scenario across every condition
def build_prompts(scenarios, conditions):
    rows = [build_prompt(scenario=scenario, condition=condition)
            for _, scenario in scenarios.iterrows() for condition in conditions]
    prompts = pd.DataFrame([row for row in rows if row])[PROMPT_COLUMNS]
    print(f'Prompts: {shape_of(prompts)}')
    return prompts


# Define function to check that every prompt ends in the request it asked for
def check_matching(prompts, scenarios):
    by_condition = {condition['name']: condition for condition in CONDITIONS}
    indexed = scenarios.set_index('scenario_id')
    unmatched = [row.prompt_id for row in prompts.itertuples()
                 if not row.prompt.endswith(
                     prompt_for(indexed.loc[row.scenario_id],
                                by_condition[row.condition]))]
    if unmatched:
        return [f'{len(unmatched)} prompts do not end in the variant their '
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
                                'expected_action': ACTIONS + ['']})
    return problems + check_matching(prompts=prompts, scenarios=scenarios)


# Define function to report what the prompt file contains
def summarise(prompts):
    signals = prompts.groupby('signal').size()
    scored = int((prompts['expected_action'] != '').sum())
    print('Signals: ' + ', '.join(f'{signal} {count}'
                                  for signal, count in signals.items()))
    print(f'Actions: {scored} scored, {len(prompts) - scored} reference')
    if len(prompts) != EXPECTED_PROMPTS:
        print(f'Expected: {EXPECTED_PROMPTS} prompts once every scenario is written')

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    section('Model prompts')
    scenarios = load_benchmark(benchmark_path=BENCHMARK_PATH)
    if scenarios.empty:
        raise SystemExit('No scenarios written yet.')
    report('benchmark.csv', check_benchmark(scenarios))

    prompts = build_prompts(scenarios=scenarios, conditions=CONDITIONS)
    report('prompts.csv', check_prompts(prompts=prompts, scenarios=scenarios))

    prompts.to_csv(PROMPTS_PATH, index=False)
    summarise(prompts=prompts)
