"""Judges five replies from each model and writes them where they can be read.

    python test_classification.py
    python test_classification.py --n 10 --backend ollama --workers 4
    python test_classification.py --model claude-haiku-4-5-20251001

Run it from the repository root. Writes to results/classification/, which is
kept apart from results/judgements/ so that a trial never mixes with the pass.

The full stage reports counts. This reports rows: the request, the reply, and
every field the classifier returned, so the verdicts can be read rather than
summarised. Five per model is enough to see whether the policy is producing what
it should, and it costs about thirty calls.

Only the classification is here. The language measures are computed by
scripts/language.py from the same replies, because they need no model and should
not cost a call to revise.

What to look for, in order.

Whether every verdict parsed. An unreadable one means the classifier ran out of
budget mid-reasoning, and the fix is a larger --tokens rather than a different
model.

Whether Refusal and Compliance are landing where you would put them.

Whether Directive Strength varies. If every reply comes back None, the measure
is doing nothing and the definition needs sharpening before forty six thousand
classifications depend on it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, 'scripts')
import evaluate
import settings
import utils

OUTPUT_DIR = settings.RESULTS_DIR / 'classification'


# Define function to draw a few replies spread across the strata, so that
# refusals and compliances are both represented rather than whichever came first
def sample_for(model, how_many, prompts, benchmark):
    collected = utils.read_lines(utils.result_path(model, settings.ADAPTATION_DIR))
    if collected.empty:
        return collected
    frame = (collected.merge(prompts[['prompt_id', 'scenario_id', 'condition',
                                      'expected_answer']], on='prompt_id')
             .merge(benchmark[['scenario_id', 'domain', 'scenario_type',
                               'request']], on='scenario_id'))
    said = frame[frame['response'].astype(str).str.strip() != '']
    if said.empty:
        return said
    per_stratum = max(1, -(-how_many // said['scenario_type'].nunique()))
    return (said.sample(frac=1, random_state=settings.SEED)
            .groupby('scenario_type', group_keys=False)
            .head(per_stratum).head(how_many))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=5, help='replies per model')
    parser.add_argument('--model', default='', help='one model, or all of them')
    parser.add_argument('--backend', default='ollama')
    parser.add_argument('--judge', default='')
    parser.add_argument('--tokens', type=int, default=0,
                        help='output budget, shared with the hidden reasoning')
    arguments = parser.parse_args()

    if arguments.tokens:
        evaluate.JUDGE_TOKENS = arguments.tokens
    judge = arguments.judge or settings.JUDGE['id']
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = utils.read_table(settings.PROMPTS_PATH)
    benchmark = utils.read_table(settings.BENCHMARK_PATH)

    wanted = ([arguments.model] if arguments.model else
              [path.stem for path in sorted(settings.ADAPTATION_DIR.glob('*.jsonl'))])
    if not wanted:
        raise SystemExit(f'Nothing collected in {settings.ADAPTATION_DIR}')

    print(f'Classifier  {judge} through {arguments.backend}')
    print(f'Policy      {len(evaluate.build_policy()) // 4:,} tokens, '
          f'budget {evaluate.JUDGE_TOKENS:,}, '
          f'temperature {evaluate.JUDGE_TEMPERATURE}')
    print(f'Writing     {OUTPUT_DIR}\n')

    unreadable = 0
    for stem in wanted:
        # the file is named after the model with the awkward characters replaced,
        # so the model itself is read back from the records
        collected = utils.read_lines(settings.ADAPTATION_DIR / f'{stem}.jsonl')
        if collected.empty:
            print(f'{stem}: nothing collected')
            continue
        model = str(collected['model'].iloc[0])
        sample = sample_for(model, arguments.n, prompts, benchmark)
        if sample.empty:
            print(f'{model}: no readable replies')
            continue

        path = OUTPUT_DIR / f'{stem}.jsonl'
        path.unlink(missing_ok=True)
        print('=' * 74)
        print(model)
        print('=' * 74)
        for row in sample.itertuples():
            started = time.time()
            verdict = evaluate.judge_reply(
                judge=judge, reply=row.response, request=row.request,
                expected=row.expected_answer, backend=arguments.backend)
            took = time.time() - started
            record = {'prompt_id': row.prompt_id, 'model': model,
                      'replicate': row.replicate, 'judge': judge, **verdict}
            utils.append_line(path, record)

            print(f'\n  {row.prompt_id}  {row.domain}, {row.scenario_type}, '
                  f'{row.condition}  ({took:.1f}s)')
            print(f'    asked    {row.request}')
            reply = str(row.response).replace('\n', ' ')
            print(f'    replied  {reply[:150]}{"..." if len(reply) > 150 else ""}')
            if verdict.get('unreadable'):
                unreadable += 1
                print(f'    UNREADABLE  {verdict["unreadable"]}')
                continue
            said = ', '.join(f'{field.replace("_", " ")}={value}'
                             for field, value in verdict.items()
                             if field not in ('unreadable', 'deviation',
                                              'expected_answer'))
            print(f'    verdict  {said}')
            print(f'    expected {row.expected_answer or "not scored"}, '
                  f'deviation {verdict.get("deviation", "")}, '
                  f'{len(str(row.response).split())} words')
        print(f'\n  written to {path}')

    print(f'\n{unreadable} verdicts could not be read.')
    if unreadable:
        print('That is the classifier running out of budget mid-reasoning, not')
        print('the policy being wrong. Raise --tokens and run it again.')
    else:
        print('Read the verdicts above before the full pass. A measure that never')
        print('varies is one the definition has not made decidable.')