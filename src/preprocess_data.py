import ast
from pathlib import Path

import pandas as pd

RAW_DIR = Path('data/raw')
CANDIDATE_DIR = Path('data/candidates')
POOL_PATH = Path('data/candidates.csv')
COVERAGE_PATH = Path('data/coverage.csv')

COLUMNS = ['candidate_id', 'source', 'source_id', 'source_licence',
           'source_category', 'category_suggested', 'stratum_hint',
           'request_form', 'needs_rewrite', 'text']

TARGET_COUNTS = {'abuse_hate': 11, 'bullying': 11, 'dangerous_challenges': 11,
                 'eating_disorders': 11, 'emotional_dependency': 11,
                 'harmful_substances': 11, 'self_harm_suicide': 12,
                 'sexual_content': 11, 'violence': 11}

CATEGORY_NAMES = {
    'abuse_hate':           'Abuse & Hate',
    'bullying':             'Bullying',
    'dangerous_challenges': 'Dangerous Challenges',
    'eating_disorders':     'Eating Disorders',
    'emotional_dependency': 'Emotional Dependency',
    'harmful_substances':   'Harmful Substances',
    'self_harm_suicide':    'Self-Harm & Suicide',
    'sexual_content':       'Sexual Content',
    'violence':             'Violence',
}

OSA_TIER = {
    'eating_disorders': 'PPC', 'self_harm_suicide': 'PPC', 'sexual_content': 'PPC',
    'abuse_hate': 'PC', 'bullying': 'PC', 'dangerous_challenges': 'PC',
    'harmful_substances': 'PC', 'violence': 'PC',
    'emotional_dependency': 'NDC',
}

INSTRUCTION_PREFIXES = ('write ', 'tell me ', 'give me ', 'explain how ', 'describe ',
                        'list ', 'show me ', 'create ', 'generate ', 'make ',
                        'draft ', 'compose ', 'provide ')

XSTEST_TYPES = ['definitions', 'safe_contexts', 'safe_targets',
                'homonyms', 'figurative_language', 'historical_events']

XSTEST_EXCLUDED_IDS = [48, 78]

MINORBENCH_MAP = {
    'Sexual':        ('sexual_content', 'age_sensitive'),
    'Self-harm':     ('self_harm_suicide', 'age_sensitive'),
    'Substance use': ('harmful_substances', 'adult_teen_only'),
    'Hateful':       ('abuse_hate', 'age_sensitive'),
    'Danger':        ('dangerous_challenges', 'age_sensitive'),
}

DONOTANSWER_MAP = {
    'Adult Content':                                ('sexual_content', 'age_sensitive'),
    'Mental Health or Overreliance Crisis':         ('emotional_dependency', 'age_sensitive'),
    'Treat Chatbot as a Human':                     ('emotional_dependency', 'age_sensitive'),
    'Toxic language (hate speech)':                 ('abuse_hate', 'age_sensitive'),
    'Social stereotypes and unfair discrimination': ('abuse_hate', 'age_sensitive'),
    'Nudging or advising users to perform unethical or unsafe actions': ('dangerous_challenges', 'age_sensitive'),
    'Causing material harm by disseminating misinformation e.g. in medicine or law': ('harmful_substances', 'age_sensitive'),
    'Compromise privacy by leaking or inferring private information (person/individual)': ('emotional_dependency', 'age_sensitive'),
}

ORBENCH_MAP = {'self-harm': 'self_harm_suicide', 'sexual': 'sexual_content',
               'violence': 'violence', 'hate': 'abuse_hate'}

SST_MAP = {
    'Suicide, Self-Harm, and Eating Disorders': ('self_harm_suicide', 'unsafe_any_age'),
    'Physical Harm and Violence':               ('violence', 'unsafe_any_age'),
    'Illegal and Highly Regulated items':       ('harmful_substances', 'adult_teen_only'),
}

SAFECHILD_MAP = {
    'bullying':  ('bullying', 'age_sensitive'),
    'self_harm': ('self_harm_suicide', 'age_sensitive'),
    'sexual':    ('sexual_content', 'age_sensitive'),
    'substance': ('harmful_substances', 'adult_teen_only'),
    'violence':  ('violence', 'age_sensitive'),
}


# Define function to classify each prompt as a question or an instruction
def classify_request_form(prompts):
    lowered = prompts.astype(str).str.strip().str.lower()
    is_instruction = lowered.str.startswith(INSTRUCTION_PREFIXES)
    return pd.Series(['instruction' if flag else 'question' for flag in is_instruction],
                     index=prompts.index)


# Define function to assemble a normalised candidate frame
def build_candidates(prefix, source, licence, ids, categories, osa, stratum, needs_rewrite, text):
    text = pd.Series(list(text))
    return pd.DataFrame({
        'candidate_id': [f'{prefix}_{i}' for i in ids],
        'source': source,
        'source_id': list(ids),
        'source_licence': licence,
        'source_category': list(categories),
        'category_suggested': list(osa),
        'stratum_hint': list(stratum),
        'request_form': list(classify_request_form(text)),
        'needs_rewrite': needs_rewrite,
        'text': list(text),
    })[COLUMNS]


# Define function to strip the contrast prefix from an XSTest type label
def base_type(types):
    return types.str.replace('contrast_', '', regex=False)


# Define function to filter XSTest safe prompts to the child-plausible types
def prepare_xstest_safe(raw_dir):
    df = pd.read_csv(raw_dir / 'xstest.csv')
    df = df[(df['label'] == 'safe') & df['type'].isin(XSTEST_TYPES)]
    return build_candidates(
        prefix='xstest', source='XSTest-safe', licence='CC BY 4.0',
        ids=df['id'], categories=df['type'], osa=['violence'] * len(df),
        stratum=['benign_control'] * len(df), needs_rewrite=False, text=df['prompt'])


# Define function to filter XSTest unsafe twins for the age-invariant stratum
def prepare_xstest_unsafe(raw_dir):
    df = pd.read_csv(raw_dir / 'xstest.csv')
    df = df[(df['label'] == 'unsafe') & ~df['id'].isin(XSTEST_EXCLUDED_IDS)]
    df = df[base_type(df['type']).isin(XSTEST_TYPES)]
    return build_candidates(
        prefix='xstest', source='XSTest-unsafe', licence='CC BY 4.0',
        ids=df['id'], categories=df['type'], osa=['violence'] * len(df),
        stratum=['unsafe_any_age'] * len(df), needs_rewrite=False, text=df['prompt'])


# Define function to match each safe prompt to its unsafe lexical twin
def prepare_xstest_pairs(raw_dir):
    df = pd.read_csv(raw_dir / 'xstest.csv')
    df = df[~df['id'].isin(XSTEST_EXCLUDED_IDS)]
    safe = df[df['label'] == 'safe']
    unsafe = df[df['label'] == 'unsafe'].assign(base_type=lambda d: base_type(d['type']))
    pairs = pd.merge(safe, unsafe, left_on=['focus', 'type'],
                     right_on=['focus', 'base_type'], suffixes=('_safe', '_unsafe'))
    pairs = pairs.drop_duplicates(subset=['focus', 'type_safe'])
    return pd.DataFrame({
        'pair_id': 'pair_' + pairs['focus'].str.replace(' ', '_'),
        'focus': pairs['focus'],
        'type': pairs['type_safe'],
        'safe_id': pairs['id_safe'],
        'safe_text': pairs['prompt_safe'],
        'unsafe_id': pairs['id_unsafe'],
        'unsafe_text': pairs['prompt_unsafe'],
    })


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
        osa=['dangerous_challenges'] * len(df), stratum=['age_sensitive'] * len(df),
        needs_rewrite=True, text=stems)


# Define function to filter SimpleSafetyTests, excluding the child abuse harm area
def prepare_simplesafetytests(raw_dir):
    df = pd.read_csv(raw_dir / 'simplesafetytests.csv')
    df = df[df['harm_area'].isin(SST_MAP)]
    return build_candidates(
        prefix='sst', source='SimpleSafetyTests', licence='CC BY-NC 4.0',
        ids=df['id'], categories=df['harm_area'] + ' | ' + df['category'],
        osa=df['harm_area'].map(lambda h: SST_MAP[h][0]),
        stratum=df['harm_area'].map(lambda h: SST_MAP[h][1]),
        needs_rewrite=False, text=df['prompt'])


# Define function to filter Safe-Child-LLM and de-age its child-framed prompts
def prepare_safechildllm(raw_dir, category_column='category', text_column='prompt'):
    path = raw_dir / 'safechildllm.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    keys = df[category_column].astype(str).str.lower().str.replace(' ', '_')
    df = df[keys.isin(SAFECHILD_MAP)]
    keys = keys[keys.isin(SAFECHILD_MAP)]
    return build_candidates(
        prefix='scl', source='Safe-Child-LLM', licence='MIT',
        ids=df.index, categories=df[category_column],
        osa=keys.map(lambda c: SAFECHILD_MAP[c][0]),
        stratum=keys.map(lambda c: SAFECHILD_MAP[c][1]),
        needs_rewrite=True, text=df[text_column])


# Define function to combine and save all candidate frames
def combine_candidates(frames, candidate_dir, pool_path):
    candidate_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: frame for name, frame in frames.items() if frame is not None}
    for name, frame in frames.items():
        frame.to_csv(candidate_dir / f'{name}.csv', index=False)
        print(f'{name}: {len(frame)} candidates retained')
    pool = pd.concat(frames.values(), ignore_index=True)
    pool.to_csv(pool_path, index=False)
    print(f'pool: {len(pool)} candidates written to {pool_path}')
    return pool


# Define function to report candidate coverage against the target design
def report_coverage(pool, target_counts):
    counts = pool['category_suggested'].value_counts()
    coverage = pd.DataFrame({
        'name': [CATEGORY_NAMES[c] for c in target_counts],
        'tier': [OSA_TIER[c] for c in target_counts],
        'available': [int(counts.get(category, 0)) for category in target_counts],
        'required': list(target_counts.values()),
    }, index=list(target_counts.keys()))
    coverage['shortfall'] = (coverage['required'] - coverage['available']).clip(lower=0)
    print(coverage.to_string())
    return coverage


# Define function to summarise licence and request form composition of the pool
def report_composition(pool):
    print(pool['source_licence'].value_counts().to_string())
    print(pool['request_form'].value_counts().to_string())
    print(f"{int(pool['needs_rewrite'].sum())} candidates require rewriting")


if __name__ == '__main__':
    candidate_frames = {
        'xstest_safe': prepare_xstest_safe(raw_dir=RAW_DIR),
        'xstest_unsafe': prepare_xstest_unsafe(raw_dir=RAW_DIR),
        'minorbench': prepare_minorbench(raw_dir=RAW_DIR),
        'donotanswer': prepare_donotanswer(raw_dir=RAW_DIR),
        'orbench': prepare_orbench(raw_dir=RAW_DIR),
        'safetext': prepare_safetext(raw_dir=RAW_DIR),
        'simplesafetytests': prepare_simplesafetytests(raw_dir=RAW_DIR),
        'safechildllm': prepare_safechildllm(raw_dir=RAW_DIR),
    }

    candidate_pool = combine_candidates(frames=candidate_frames,
                                        candidate_dir=CANDIDATE_DIR,
                                        pool_path=POOL_PATH)

    coverage_table = report_coverage(pool=candidate_pool, target_counts=TARGET_COUNTS)
    coverage_table.to_csv(COVERAGE_PATH)
    report_composition(pool=candidate_pool)

    xstest_pairs = prepare_xstest_pairs(raw_dir=RAW_DIR)
    xstest_pairs.to_csv(CANDIDATE_DIR / 'xstest_pairs.csv', index=False)
    print(f'xstest pairs: {len(xstest_pairs)} matched safe and unsafe twins')