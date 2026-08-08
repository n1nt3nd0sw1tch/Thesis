"""Traces the lexical simplification of one request, word by word, so that
every decision the simplifier makes can be inspected before it is trusted.
Pass a request on the command line, or leave it out to use the first written
one in the benchmark."""

import sys
import pandas as pd
from nltk import pos_tag, word_tokenize
from wordfreq import zipf_frequency
from settings import BENCHMARK_PATH, section, written
from simplify_data import (AOA_PATH, MINIMUM_GAIN, PROTECTED, TARGET_AGES,
                           age_of, lemma_of, load_aoa, simplify, synonyms)

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to read the request being traced
def read_request(benchmark_path):
    if len(sys.argv) > 1:
        return ' '.join(sys.argv[1:])
    benchmark = written(pd.read_csv(benchmark_path, dtype=str).fillna(''))
    if benchmark.empty:
        raise SystemExit('No scenarios written yet, pass a request instead.')
    return benchmark['request'].iloc[0]


# Define function to show the age of acquisition of every word
def show_words(request, aoa, target_age):
    rows = []
    for word, tag in pos_tag(word_tokenize(request)):
        if not word.isalpha():
            continue
        lemma = lemma_of(word, tag)
        rating = age_of(word, tag, aoa)
        if lemma in PROTECTED or word.lower() in PROTECTED:
            verdict = 'protected'
        elif rating is None:
            verdict = 'not in norms'
        elif rating <= target_age:
            verdict = 'already simple'
        else:
            verdict = 'candidate'
        rows.append({'word': word, 'tag': tag, 'lemma': lemma,
                     'aoa': rating, 'zipf': round(zipf_frequency(lemma, 'en'), 2),
                     'verdict': verdict})
    frame = pd.DataFrame(rows)
    section(f'Words, against a target age of {target_age}')
    print(frame.to_string(index=False))
    return frame


# Define function to show why each candidate was or was not replaced
def show_candidates(frame, aoa, target_age):
    section('Synonyms considered')
    for row in frame[frame['verdict'] == 'candidate'].itertuples():
        print(f'\n{row.word} (aoa {row.aoa}, zipf {row.zipf})')
        offered = synonyms(row.lemma, row.tag)
        if not offered:
            print('  wordnet offers nothing')
            continue
        for candidate in sorted(offered):
            rating = aoa.get(candidate)
            zipf = round(zipf_frequency(candidate, 'en'), 2)
            if rating is None:
                reason = 'not in norms'
            elif rating > target_age:
                reason = f'still above {target_age}'
            elif row.aoa - rating < MINIMUM_GAIN:
                reason = f'gain only {round(row.aoa - rating, 2)}'
            elif zipf < row.zipf:
                reason = 'rarer than the original'
            else:
                reason = 'ACCEPTED'
            print(f'  {candidate:<18} aoa {str(rating):<6} zipf {zipf:<5} {reason}')


# Define function to show the variant produced for every target age
def show_variants(request, aoa, target_ages):
    section('Variant per target age')
    for age in target_ages:
        rebuilt, changes = simplify(request, aoa, age)
        marker = ' '.join(changes) if changes else 'unchanged'
        print(f'{age:>3}  {marker}')
        print(f'     {rebuilt}')

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    request = read_request(BENCHMARK_PATH)
    norms = load_aoa(AOA_PATH)
    section('Request')
    print(request)
    print(f'Norms: {len(norms)} words')

    words = show_words(request, norms, TARGET_AGES[0])
    show_candidates(words, norms, TARGET_AGES[0])
    show_variants(request, norms, TARGET_AGES)