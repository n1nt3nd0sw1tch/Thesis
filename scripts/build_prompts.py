"""Expands every written scenario across the age conditions into the model-ready
prompt file. Each condition contributes two turns: the age signal and the shared
request. Turn 2 is identical across every condition of a scenario, which is
checked before the file is written."""

import pandas as pd

from settings import (ACTIONS, CONDITION_AGES, CONDITION_NAMES, CONDITIONS,
                      PROMPT_COLUMNS, PROMPTS_PATH, SCENARIOS_PATH, SIGNALS,
                      TOTAL_SCENARIOS, TURNS, check_scenarios, make_prompt_id,
                      preview, report, validate, written)

# =============================================================================
# Settings
# =============================================================================

# The conditions themselves live in config/benchmark.yml. Nothing here needs
# editing.
EXPECTED_ROWS = TOTAL_SCENARIOS * len(CONDITIONS) * len(TURNS)

# =============================================================================
# Functions
# =============================================================================


# Define function to read the scenario worksheet and keep only written rows
def load_scenarios(scenarios_path):
    if not scenarios_path.exists():
        raise FileNotFoundError('scenarios.csv not found, run prepare_data.py first')
    scenarios = pd.read_csv(scenarios_path, dtype=str).fillna('')
    rows = written(scenarios)
    print(f'Scenarios written: {len(rows)} of {len(scenarios)}')
    return rows


# Define function to build the two prompt rows one condition contributes
def build_exchange(scenario, condition):
    band = condition['band']
    common = {'scenario_id': scenario['scenario_id'], 'condition': condition['name'],
              'age': condition['age'], 'signal': condition['signal']}
    return [
        {'prompt_id': make_prompt_id(scenario['scenario_id'], condition['name'], 1),
         **common, 'turn': 1, 'prompt': condition['opener'], 'expected_action': ''},
        {'prompt_id': make_prompt_id(scenario['scenario_id'], condition['name'], 2),
         **common, 'turn': 2, 'prompt': scenario['request'],
         'expected_action': scenario[band] if band else ''},
    ]


# Define function to expand every scenario across every condition
def build_prompts(scenarios, conditions):
    rows = [row for _, scenario in scenarios.iterrows()
            for condition in conditions
            for row in build_exchange(scenario=scenario, condition=condition)]
    return pd.DataFrame(rows)[PROMPT_COLUMNS]


# Define function to check that the request never varies by condition
def check_requests(prompts):
    requests = prompts[prompts['turn'] == 2]
    varying = requests.groupby('scenario_id')['prompt'].nunique()
    varying = varying[varying > 1]
    if len(varying):
        return [f'prompts.csv: turn 2 differs across conditions for '
                f'{", ".join(varying.index[:5])}']
    counts = requests.groupby('scenario_id').size()
    wrong = counts[counts != len(CONDITIONS)]
    if len(wrong):
        return [f'prompts.csv: {len(wrong)} scenarios do not have '
                f'{len(CONDITIONS)} conditions']
    return []


# Define function to check the expanded prompt file
def check_prompts(prompts):
    problems = validate(frame=prompts, name='prompts.csv',
                        required=PROMPT_COLUMNS, id_column='prompt_id',
                        text_columns=['prompt_id', 'scenario_id', 'prompt'],
                        labels={'condition': CONDITION_NAMES, 'age': CONDITION_AGES,
                                'signal': SIGNALS, 'turn': TURNS,
                                'expected_action': ACTIONS + ['']})
    return problems + check_requests(prompts=prompts)


# Define function to report the shape of the file that was written
def summarise(prompts):
    requests = prompts[prompts['turn'] == 2]
    counts = requests.groupby('signal').size()
    scored = int((requests['expected_action'] != '').sum())
    print(f'\n{len(prompts)} rows, {prompts["scenario_id"].nunique()} scenarios, '
          f'{len(CONDITIONS)} conditions, {len(TURNS)} turns')
    print(', '.join(f'{signal} {count}' for signal, count in counts.items()))
    print(f'{scored} scored turns, {len(requests) - scored} reference turns')
    if len(prompts) != EXPECTED_ROWS:
        print(f'note: {EXPECTED_ROWS} rows expected once all scenarios are written')


# =============================================================================
# Run
# =============================================================================

if __name__ == '__main__':
    scenarios = load_scenarios(scenarios_path=SCENARIOS_PATH)
    if scenarios.empty:
        raise SystemExit('No scenarios written yet.')
    report(check_scenarios(scenarios))

    prompts = build_prompts(scenarios=scenarios, conditions=CONDITIONS)
    report(check_prompts(prompts=prompts))

    prompts.to_csv(PROMPTS_PATH, index=False)

    preview(prompts, 'Prompts')
    summarise(prompts=prompts)
