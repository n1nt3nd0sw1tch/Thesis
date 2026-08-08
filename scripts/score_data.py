"""Scores text for readability and lexical difficulty. Used first to check that
each implicit variant differs from the neutral prompt in its cue alone, and later on
the model responses, where the same measures answer whether a model adapts its
language to the age it was given."""

import pandas as pd
import textstat
from nltk import pos_tag, word_tokenize
from settings import (AGE_BANDS, BENCHMARK_PATH, SCORE_COLUMNS, SCORES_PATH,
                      report, section, shape_of, validate, written)
from simplify_data import AOA_PATH, age_of, load_aoa

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

# A word counts as difficult for a reader of this age when it is acquired later.
DIFFICULT_ABOVE = 10

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

# Define function to list the age of acquisition of every word in a text
def ratings_of(text, aoa):
    ratings = [age_of(word, tag, aoa)
               for word, tag in pos_tag(word_tokenize(str(text))) if word.isalpha()]
    return [rating for rating in ratings if rating is not None]


# Define function to score one text for readability and lexical difficulty
def score(text, aoa):
    ratings = ratings_of(text, aoa)
    words = [word for word in word_tokenize(str(text)) if word.isalpha()]
    return {
        'words': len(words),
        'fkgl': round(textstat.flesch_kincaid_grade(str(text)), 2),
        'fre': round(textstat.flesch_reading_ease(str(text)), 2),
        'mean_aoa': round(sum(ratings) / len(ratings), 2) if ratings else 0.0,
        'max_aoa': round(max(ratings), 2) if ratings else 0.0,
        'difficult': sum(1 for rating in ratings if rating > DIFFICULT_ABOVE),
        'covered': round(len(ratings) / len(words), 2) if words else 0.0,
    }


# Define function to score every text in one column of a table
def score_frame(frame, name, aoa, key='scenario_id', column='prompt'):
    rows = [{'variant': name, key: getattr(row, key), **score(getattr(row, column), aoa)}
            for row in frame.itertuples()]
    return pd.DataFrame(rows)


# Define function to compare the variants on every measure
def report_comparison(scores):
    measures = ['words', 'fkgl', 'fre', 'mean_aoa', 'max_aoa', 'difficult', 'covered']
    section('Mean score per variant')
    print(scores.groupby('variant')[measures].mean().round(2).to_string())

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    section('Scores')
    norms = load_aoa(AOA_PATH)

    benchmark = written(pd.read_csv(BENCHMARK_PATH, dtype=str).fillna(''))
    if benchmark.empty:
        raise SystemExit('No scenarios written yet.')
    frames = [score_frame(benchmark, 'neutral', norms)]
    for band in AGE_BANDS:
        variants = benchmark[benchmark[f'implicit_{band}'].str.strip() != '']
        if not variants.empty:
            frames.append(score_frame(variants, f'implicit_{band}', norms,
                                      column=f'implicit_{band}'))


    scores = pd.concat(frames, ignore_index=True)[SCORE_COLUMNS]
    print(f'Scores: {shape_of(scores)}')
    report('scores.csv', validate(frame=scores, required=SCORE_COLUMNS,
                                  text_columns=['variant', 'scenario_id']))

    scores.to_csv(SCORES_PATH, index=False)
    report_comparison(scores=scores)
