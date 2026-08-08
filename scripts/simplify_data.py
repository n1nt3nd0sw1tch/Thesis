"""Builds lexically simplified variants of every written request. A word is
replaced only when its age of acquisition is above the target age, a WordNet
synonym of the same part of speech is acquired earlier and used at least as
often, and the word is not one the scenario depends on. No language model is
involved, so every substitution is traceable to the norms it came from."""

import pandas as pd
from lemminflect import getInflection, getLemma
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from wordfreq import zipf_frequency
from settings import (BENCHMARK_PATH, LEXICAL_COLUMNS, LEXICAL_PATH, ORIGINAL_DIR,
                      report, section, shape_of, validate, written)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

# Kuperman, Stadthagen-Gonzalez and Brysbaert (2012), Behavior Research Methods
# 44(4), downloaded by download_data.py and cited in the write-up.
AOA_PATH = ORIGINAL_DIR / 'aoa.csv'
AOA_WORD_COLUMNS = ['Word', 'word']
AOA_RATING_COLUMNS = ['Rating.Mean', 'AoA_Kup_lem', 'AoA_Kup', 'aoa']

# One variant per age band. A word is a candidate when its age of acquisition
# is above the target age.
TARGET_AGES = [7, 9, 11, 13, 15, 17]

# A substitution must lower the age of acquisition by at least this much, and
# must not make the word rarer. Zipf frequency is on a log scale where 4.0 is
# roughly the ten thousandth commonest word.
MINIMUM_GAIN = 1.0
MAXIMUM_SUBSTITUTIONS = 3

# Words the scenario depends on. Replacing any of these changes what is being
# asked rather than how difficult it is to read.
PROTECTED = {
    'alcohol', 'anorexia', 'bulimia', 'cannabis', 'cocaine', 'drug', 'drugs',
    'drunk', 'ecstasy', 'gun', 'heroin', 'kill', 'knife', 'marijuana', 'meth',
    'nude', 'overdose', 'pill', 'pills', 'porn', 'pornography', 'purge',
    'sex', 'sexual', 'starve', 'suicide', 'vape', 'vodka', 'weapon', 'weed',
}

# Penn tags mapped onto the four parts of speech WordNet distinguishes
WORDNET_TAGS = {'J': wordnet.ADJ, 'N': wordnet.NOUN,
                'R': wordnet.ADV, 'V': wordnet.VERB}

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to read the age of acquisition norms
def load_aoa(path):
    if not path.exists():
        raise FileNotFoundError(
            f'{path.name} not found, run download_data.py first')
    frame = pd.read_csv(path)
    word = next(c for c in AOA_WORD_COLUMNS if c in frame.columns)
    rating = next(c for c in AOA_RATING_COLUMNS if c in frame.columns)
    frame = frame[[word, rating]].dropna()
    return dict(zip(frame[word].astype(str).str.lower(),
                    pd.to_numeric(frame[rating], errors='coerce')))


# Define function to reduce a word to the lemma the norms are keyed on
def lemma_of(word, tag):
    lemmas = getLemma(word.lower(), upos={'J': 'ADJ', 'N': 'NOUN', 'R': 'ADV',
                                          'V': 'VERB'}.get(tag[0], 'NOUN'))
    return lemmas[0].lower() if lemmas else word.lower()


# Define function to look up the age of acquisition of one word
def age_of(word, tag, aoa):
    return aoa.get(word.lower(), aoa.get(lemma_of(word, tag)))


# Define function to collect the single word synonyms WordNet offers
def synonyms(lemma, tag):
    part = WORDNET_TAGS.get(tag[0])
    if part is None:
        return []
    found = {name.name().lower() for synset in wordnet.synsets(lemma, part)
             for name in synset.lemmas()}
    return [word for word in found if word.isalpha() and word != lemma]


# Define function to choose the earliest acquired synonym worth substituting
def best_synonym(word, tag, aoa, target_age):
    lemma = lemma_of(word, tag)
    if lemma in PROTECTED or word.lower() in PROTECTED:
        return None
    original = age_of(word, tag, aoa)
    if original is None or original <= target_age:
        return None

    ranked = []
    for candidate in synonyms(lemma, tag):
        rating = aoa.get(candidate)
        if rating is None or rating > target_age or original - rating < MINIMUM_GAIN:
            continue
        if zipf_frequency(candidate, 'en') < zipf_frequency(lemma, 'en'):
            continue
        ranked.append((rating, -zipf_frequency(candidate, 'en'), candidate))
    if not ranked:
        return None

    replacement = min(ranked)[2]
    inflected = getInflection(replacement, tag=tag)
    return inflected[0] if inflected else replacement


# Define function to match the capitalisation of the word being replaced
def match_case(original, replacement):
    return replacement.capitalize() if original[:1].isupper() else replacement


# Define function to rewrite one request for one target age
def simplify(request, aoa, target_age):
    words = word_tokenize(request)
    tagged = pos_tag(words)
    changes = []
    for position, (word, tag) in enumerate(tagged):
        if len(changes) == MAXIMUM_SUBSTITUTIONS or not word.isalpha():
            continue
        replacement = best_synonym(word, tag, aoa, target_age)
        if replacement:
            changes.append(f'{word.lower()}>{replacement.lower()}')
            words[position] = match_case(word, replacement)
    rebuilt = ' '.join(words)
    for mark in [' .', ' ,', ' ?', ' !', " '", ' n’t', " n't"]:
        rebuilt = rebuilt.replace(mark, mark.strip())
    return rebuilt, changes


# Define function to describe how difficult the wording of a request is
def describe(request, aoa):
    ratings = [age_of(word, tag, aoa) for word, tag in pos_tag(word_tokenize(request))
               if word.isalpha()]
    ratings = [rating for rating in ratings if rating is not None]
    if not ratings:
        return 0.0, 0.0
    return round(sum(ratings) / len(ratings), 2), round(max(ratings), 2)


# Define function to build every variant of every written request
def build_variants(benchmark, aoa, target_ages):
    rows = []
    for scenario in benchmark.itertuples():
        for age in target_ages:
            request, changes = simplify(scenario.request, aoa, age)
            mean_aoa, max_aoa = describe(request, aoa)
            rows.append({'scenario_id': scenario.scenario_id, 'age': age,
                         'request': request, 'mean_aoa': mean_aoa,
                         'max_aoa': max_aoa, 'changes': ' '.join(changes),
                         'n_changes': len(changes)})
    return pd.DataFrame(rows)[LEXICAL_COLUMNS]


# Define function to report how much of the benchmark could be simplified
def report_coverage(variants):
    coverage = variants.groupby('age').agg(
        simplified=('n_changes', lambda counts: int((counts > 0).sum())),
        scenarios=('n_changes', 'size'),
        substitutions=('n_changes', 'sum'),
        mean_aoa=('mean_aoa', 'mean')).round(2)
    coverage['share'] = (coverage['simplified'] / coverage['scenarios']).round(2)
    section('Coverage by target age')
    print(coverage.to_string())

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    section('Lexical variants')
    norms = load_aoa(AOA_PATH)
    print(f'Norms: {len(norms)} words')

    benchmark = written(pd.read_csv(BENCHMARK_PATH, dtype=str).fillna(''))
    if benchmark.empty:
        raise SystemExit('No scenarios written yet.')

    variants = build_variants(benchmark=benchmark, aoa=norms, target_ages=TARGET_AGES)
    print(f'Variants: {shape_of(variants)}')
    report('lexical.csv', validate(frame=variants, required=LEXICAL_COLUMNS,
                                   text_columns=['scenario_id', 'request']))

    variants.to_csv(LEXICAL_PATH, index=False)
    report_coverage(variants=variants)