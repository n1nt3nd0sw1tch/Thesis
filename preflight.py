"""Checks everything that must be right before a final batch is submitted.

    python preflight.py gpt-5.6-luna

Run it from the repository root. It changes nothing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, 'scripts')
import backends
import run
import settings
import utils

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'gpt-5.6-luna'
ok = True


def check(label, passed, detail=''):
    global ok
    ok = ok and passed
    print(f"  {'PASS' if passed else 'FAIL'}  {label}" + (f'  {detail}' if detail else ''))


print(f'Preflight for {MODEL}\n')

print('Scripts')
for name, attr in [('run', 'write_batch'), ('run', 'read_batch'),
                   ('run', 'set_aside_replies'), ('run', 'name_after_job'),
                   ('backends', 'spent'), ('settings', 'BATCHES_DIR')]:
    check(f'{name}.{attr}', hasattr(globals()[name], attr))

print('\nSampling')
provider = backends.provider_of(MODEL)
probe = backends.build_payload(provider, MODEL, [{'role': 'user', 'content': 'x'}],
                               settings.GENERATION['max_tokens'],
                               settings.GENERATION['temperature'])
takes = backends.takes_sampling(MODEL)
sent = 'temperature' in json.dumps(probe)
check('sampling matches what the panel declares', takes == sent,
      f"declared {'accepts' if takes else 'provider defaults'}, "
      f"payload {'sends' if sent else 'omits'}")
if not takes:
    print(f"        {MODEL} rejects temperature and top_p, so it runs at the "
          f"provider's own decoding")
check('reasoning left to the provider', 'reasoning' not in probe,
      probe.get('reasoning', 'no key sent'))
# each provider names the cap differently, and Google nests it
cap = (probe.get('max_output_tokens') or probe.get('max_tokens')
       or (probe.get('generationConfig') or {}).get('maxOutputTokens'))
check(f"cap is {settings.GENERATION['max_tokens']}",
      cap == settings.GENERATION['max_tokens'])

print('\nWhat is on disk')
collected = utils.read_lines(utils.result_path(MODEL, settings.ADAPTATION_DIR))
aside = sorted((settings.ADAPTATION_DIR.parent / 'superseded')
               .glob(f'{utils.model_slug(MODEL)}*.jsonl'))
print(f'  results/adaptation  {len(collected):,} replies')
for path in aside:
    print(f'  superseded          {len(utils.read_lines(path)):,} replies  {path.name}')
stale = []
for path in sorted(settings.BATCHES_DIR.glob('*_requests.jsonl')):
    line = json.loads(path.read_text().splitlines()[0])
    body = line.get('body') or line.get('params') or {}
    if body.get('model') != MODEL:
        continue
    print(f"  request file        {sum(1 for _ in path.open()):,} requests  {path.name}"
          f"\n                      temp {body.get('temperature', 'provider default')}, "
          f"top_p {body.get('top_p', 'provider default')}, "
          f"reasoning {body.get('reasoning', {}).get('effort', 'default')}")
    # a file written before a settings change still carries the old parameters,
    # and resubmitting it repeats whatever went wrong last time
    if set(body) != set(probe) or body.get('temperature') != probe.get('temperature'):
        stale.append(path.name)

if stale:
    print()
    for name in stale:
        print(f'  STALE  {name} does not match what the code would write now')
    print('         Re-export before submitting, do not reuse these files.')

print('\nWhat the next export would write')
prompts = utils.read_table(settings.PROMPTS_PATH)
wanted = len(prompts) * settings.GENERATION['replicates']
outstanding = wanted - len(collected)
print(f'  {outstanding:,} of {wanted:,} outstanding')
if outstanding == 0:
    print('  Nothing would be written. Set FRESH = True to rerun from scratch.')

print(f"\n{'Ready to submit' if ok else 'NOT ready, fix the FAIL lines above'}")