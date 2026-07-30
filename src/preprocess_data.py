import ast
from pathlib import Path

import pandas as pd

RAW_DIR = Path('data/raw')
CANDIDATE_DIR = Path('data/candidates')
POOL_PATH = Path('data/candidates.csv')

COLUMNS = ['candidate_id', 'source', 'source_id', 'source_licence',
           'source_category', 'osa_suggested', 'stratum_hint',
           'needs_rewrite', 'text']

TARGET_COUNTS = {'P1': 11, 'P2': 8, 'P3': 8, 'P4': 8, 'C1': 8, 'C2': 8,
                 'C3': 8, 'C4': 10, 'C5': 13, 'N1': 14, 'B0': 4}

XSTEST_TYPES = ['definitions', 'safe_contexts', 'safe_targets',
                'homonyms', 'figurative_language', 'historical_events']

MINORBENCH_MAP = {
    'Sexual':        ('P1', 'age_sensitive'),
    'Self-harm':     ('P3', 'age_sensitive'),
    'Substance use': ('C5', 'age_restricted'),
    'Hateful':       ('C2', 'age_sensitive'),
    'Danger':        ('C4', 'age_sensitive'),
}

DONOTANSWER_MAP = {
    'Adult Content':                                ('P1', 'age_sensitive'),
    'Mental Health or Overreliance Crisis':         ('N1', 'age_sensitive'),
    'Treat Chatbot as a Human':                     ('N1', 'age_sensitive'),
    'Toxic language (hate speech)':                 ('C2', 'age_sensitive'),
    'Social stereotypes and unfair discrimination': ('C2', 'age_sensitive'),
    'Nudging or advising users to perform unethical or unsafe actions': ('C4', 'age_sensitive'),
    'Causing material harm by disseminating misinformation e.g. in medicine or law': ('C5', 'age_sensitive'),
    'Compromise privacy by leaking or inferring private information (person/individual)': ('N1', 'age_sensitive'),
}

ORBENCH_MAP = {'self-harm': 'P3', 'sexual': 'P1', 'violence': 'C3', 'hate': 'C2'}


# Define function to assemble a normalised candidate frame
def build_candidates(prefix, source, licence, ids, categories, osa, stratum, needs_rewrite, text):
    return pd.DataFrame({
        'candidate_id': [f'{prefix}_{i}' for i in ids],
        'source': source,
        'source_id': list(ids),
        'source_licence': licence,
        'source_category': list(categories),
        'osa_suggested': list(osa),
        'stratum_hint': list(stratum),
        'needs_rewrite': needs_rewrite,
        'text': list(text),
    })[COLUMNS]


# Define function to filter XSTest to the six child-plausible safe types
def prepare_xstest(raw_dir):
    df = pd.read_csv(raw_dir / 'xstest_safe.csv')
    df = df[df['type'].isin(XSTEST_TYPES)]
    return build_candidates(
        prefix='xstest', source='XSTest', licence='CC BY 4.0',
        ids=df['id'], categories=df['type'], osa=['B0'] * len(df),
        stratum=['benign_control'] * len(df), needs_rewrite=False, text=df['prompt'])


# Define function to filter MinorBench and map its categories to OSA codes
def prepare_minorbench(raw_dir):
    df = pd.read_csv(raw_dir / 'minorbench.csv')
    df = df[df['Category'].isin(MINORBENCH_MAP)]
    return build_candidates(
        prefix='minor', source='MinorBench', licence='MIT',
        ids=df['PromptID'], categories=df['Category'],
        osa=df['Category'].map(lambda c: MINORBENCH_MAP[c][0]),
        stratum=df['Category'].map(lambda c: MINORBENCH_MAP[c][1]),
        needs_rewrite=True, text=df['Prompt'])


# Define function to filter DoNotAnswer and map its harm types to OSA codes
def prepare_donotanswer(raw_dir):
    df = pd.read_csv(raw_dir / 'donotanswer.csv')
    df = df[df['types_of_harm'].isin(DONOTANSWER_MAP)]
    return build_candidates(
        prefix='dna', source='DoNotAnswer', licence='Apache 2.0',
        ids=df['id'], categories=df['types_of_harm'],
        osa=df['types_of_harm'].map(lambda c: DONOTANSWER_MAP[c][0]),
        stratum=df['types_of_harm'].map(lambda c: DONOTANSWER_MAP[c][1]),
        needs_rewrite=False, text=df['question'])


# Define function to filter OR-Bench to categories overlapping OSA harm areas
def prepare_orbench(raw_dir):
    df = pd.read_csv(raw_dir / 'orbench_hard.csv')
    df = df[df['category'].isin(ORBENCH_MAP)]
    return build_candidates(
        prefix='orb', source='OR-Bench-hard', licence='CC BY 4.0',
        ids=df.index, categories=df['category'],
        osa=df['category'].map(ORBENCH_MAP),
        stratum=['benign_control'] * len(df), needs_rewrite=False, text=df['prompt'])


# Define function to extract the first continuation from a SafeText option list
def extract_first_option(option):
    try:
        parsed = ast.literal_eval(option) if isinstance(option, str) else option
    except (ValueError, SyntaxError):
        return str(option)
    return parsed[0] if isinstance(parsed, (list, tuple)) and parsed else str(option)


# Define function to prepare SafeText scenario stems for later rewriting
def prepare_safetext(raw_dir):
    df = pd.read_csv(raw_dir / 'safetext.csv')
    stems = (df['prompt'].str.rstrip(', ') + ' [unsafe continuation: '
             + df['unsafe_options'].map(extract_first_option) + ']')
    return build_candidates(
        prefix='safetext', source='SafeText', licence='MIT',
        ids=df.index, categories=['physical_safety'] * len(df),
        osa=['C4'] * len(df), stratum=['age_sensitive'] * len(df),
        needs_rewrite=True, text=stems)


# Define function to combine and save all candidate frames
def combine_candidates(frames, candidate_dir, pool_path):
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(candidate_dir / f'{name}.csv', index=False)
        print(f'{name}: {len(frame)} candidates retained')
    pool = pd.concat(frames.values(), ignore_index=True)
    pool.to_csv(pool_path, index=False)
    print(f'pool: {len(pool)} candidates written to {pool_path}')
    return pool


# Define function to report candidate coverage against the target design
def report_coverage(pool, target_counts):
    counts = pool['osa_suggested'].value_counts()
    coverage = pd.DataFrame({
        'available': [int(counts.get(category, 0)) for category in target_counts],
        'required': list(target_counts.values()),
    }, index=list(target_counts.keys()))
    coverage['shortfall'] = (coverage['required'] - coverage['available']).clip(lower=0)
    print(coverage.to_string())
    print(f"{int(pool['needs_rewrite'].sum())} candidates require rewriting")
    return coverage


if __name__ == '__main__':
    candidate_frames = {
        'xstest': prepare_xstest(raw_dir=RAW_DIR),
        'minorbench': prepare_minorbench(raw_dir=RAW_DIR),
        'donotanswer': prepare_donotanswer(raw_dir=RAW_DIR),
        'orbench': prepare_orbench(raw_dir=RAW_DIR),
        'safetext': prepare_safetext(raw_dir=RAW_DIR),
    }

    candidate_pool = combine_candidates(frames=candidate_frames,
                                        candidate_dir=CANDIDATE_DIR,
                                        pool_path=POOL_PATH)
    coverage_table = report_coverage(pool=candidate_pool, target_counts=TARGET_COUNTS)
    coverage_table.to_csv('data/coverage.csv')
