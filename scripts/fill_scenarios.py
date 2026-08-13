"""Writes the hand-written scenarios into drafts.csv.

A scenario is specified in config/scenarios.yml as a base and a cue clause. The base is
the canonical request and the clause is appended to make each variant, so a
variant is the control plus one phrase and the four texts of a scenario differ in
exactly one contiguous span. Building them here rather than editing the four
columns by hand is what keeps that property true after a revision.

Where a scenario derives from a source record, it is written into that record's
row, so the derivation is stored rather than asserted. Where it does not, a row
is added with the dataset left blank.
"""

import pandas as pd
from settings import DOMAIN_CODES, DRAFTS_COLUMNS, TYPES, WHO

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to build the four texts of one scenario from its base
def build_texts(base, clause):
    return {'request': f'{base}?',
            **{column: f'{base} {clause.format(who=who)}?'
               for column, who in WHO.items()}}


# Define function to write every scenario into the drafts
def fill(drafts, scenarios):
    # rows added on a previous run are dropped first, so running twice gives the
    # same file as running once and a removed scenario leaves nothing behind
    drafts = drafts[~drafts['source_id'].str.startswith('authored-')].copy()
    written, added = 0, []
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
                    written += 1
                else:
                    code = DOMAIN_CODES[domain]
                    added.append({'source_id': f'authored-{code}-{number}',
                                  'dataset': '', 'source_prompt': '', **values})
    filled = pd.concat([drafts, pd.DataFrame(added)], ignore_index=True)
    return filled[DRAFTS_COLUMNS], written, len(added)


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
