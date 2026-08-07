"""Expands every written scenario across the age conditions into the model-ready
prompt file. Each prompt is a single turn: the condition opener carries the age
signal and is followed by the scenario request. The request is identical
across every condition, which is checked before the file is written."""

import pandas as pd
from settings import (ACTIONS, AGE_BANDS, CONDITION_AGES, CONDITION_NAMES,
                      CONDITIONS, PROMPT_COLUMNS, PROMPTS_PATH, SCENARIOS_PATH,
                      SIGNALS, TOTAL_SCENARIOS, check_scenarios, make_prompt,
                      make_prompt_id, report, section, shape_of, validate,
                      written)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

EXPECTED_PROMPTS = TOTAL_SCENARIOS * len(CONDITIONS)

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to read the scenario worksheet and keep only written rows
def load_scenarios(scenarios_path):
    if not scenarios_path.exists():
        raise FileNotFoundError('scenarios.csv not found, run prepare_data.py first')
    scenarios = pd.read_csv(scenarios_path, dtype=str).fillna('')
    rows = written(scenarios)
    print(f'Scenarios written: {len(rows)} of {len(scenarios)}')
    return rows


# Define function to build the prompt one condition contributes
def build_prompt(scenario, condition):
    band = condition['band']
    return {
        'prompt_id': make_prompt_id(scenario['scenario_id'], condition['name']),
        'scenario_id': scenario['scenario_id'],
        'condition': condition['name'],
        'signal': condition['signal'],
        'band': band,
        'age': condition['age'],
        'prompt': make_prompt(condition['opener'], scenario['request']),
        'expected_action': scenario[band] if band else '',
    }


# Define function to expand every scenario across every condition
def build_prompts(scenarios, conditions):
    rows = [build_prompt(scenario=scenario, condition=condition)
            for _, scenario in scenarios.iterrows() for condition in conditions]
    prompts = pd.DataFrame(rows)[PROMPT_COLUMNS]
    print(f'Prompts: {shape_of(prompts)}')
    return prompts


# Define function to check that every prompt ends in the scenario request
def check_matching(prompts, scenarios):
    requests = scenarios.set_index('scenario_id')['request']
    matched = [prompt.endswith(requests[scenario_id]) for scenario_id, prompt
               in zip(prompts['scenario_id'], prompts['prompt'])]
    unmatched = prompts.loc[[not match for match in matched], 'prompt_id']
    if len(unmatched):
        return [f'{len(unmatched)} prompts do not end in the scenario request, '
                f'first {unmatched.iloc[0]}']
    counts = prompts.groupby('scenario_id').size()
    incomplete = counts[counts != len(CONDITIONS)]
    if len(incomplete):
        return [f'{len(incomplete)} scenarios do not have {len(CONDITIONS)} conditions']
    return []


# Define function to check the expanded prompt file
def check_prompts(prompts, scenarios):
    problems = validate(frame=prompts, required=PROMPT_COLUMNS,
                        id_column='prompt_id',
                        text_columns=['prompt_id', 'scenario_id', 'prompt'],
                        labels={'condition': CONDITION_NAMES, 'signal': SIGNALS,
                                'band': AGE_BANDS + [''], 'age': CONDITION_AGES,
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
    scenarios = load_scenarios(scenarios_path=SCENARIOS_PATH)
    if scenarios.empty:
        raise SystemExit('No scenarios written yet.')
    report('scenarios.csv', check_scenarios(scenarios))

    prompts = build_prompts(scenarios=scenarios, conditions=CONDITIONS)
    report('prompts.csv', check_prompts(prompts=prompts, scenarios=scenarios))

    prompts.to_csv(PROMPTS_PATH, index=False)
    summarise(prompts=prompts)
