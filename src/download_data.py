import csv
from datetime import date
from pathlib import Path

import pandas as pd
from datasets import load_dataset

RAW_DIR = Path('data/raw')
PROVENANCE_PATH = Path('data/provenance.csv')

SOURCES = {
    'xstest':       {'hub_id': 'Paul/XSTest',          'config': None,               'split': 'train', 'licence': 'CC BY 4.0'},
    'donotanswer':  {'hub_id': 'LibrAI/do-not-answer', 'config': None,               'split': 'train', 'licence': 'Apache 2.0'},
    'minorbench':   {'hub_id': 'govtech/MinorBench',   'config': None,               'split': 'test',  'licence': 'MIT'},
    'orbench_hard': {'hub_id': 'bench-llm/or-bench',   'config': 'or-bench-hard-1k', 'split': 'train', 'licence': 'CC BY 4.0'},
    'safetext':     {'hub_id': 'walledai/SafeText',    'config': None,               'split': 'train', 'licence': 'MIT'},
}


# Define function to download a single dataset from the Hugging Face Hub
def download_dataset(key, spec, output_dir):
    output_path = output_dir / f'{key}.csv'
    if output_path.exists():
        print(f'{key}: already present at {output_path}')
        return None
    dataset = load_dataset(spec['hub_id'], spec['config'], split=spec['split'])
    dataset.to_pandas().to_csv(output_path, index=False)
    print(f'{key}: {len(dataset)} rows written to {output_path}')
    return {
        'key': key,
        'hub_id': spec['hub_id'],
        'config': spec['config'] or '',
        'split': spec['split'],
        'licence': spec['licence'],
        'rows': len(dataset),
        'columns': ';'.join(dataset.column_names),
        'downloaded': date.today().isoformat(),
    }


# Define function to download every configured dataset
def download_all(sources, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [record for record in
            (download_dataset(key, spec, output_dir) for key, spec in sources.items())
            if record is not None]


# Define function to separate the safe and unsafe halves of XSTest
def split_xstest(raw_dir):
    df = pd.read_csv(raw_dir / 'xstest.csv')
    safe = df[df['label'] == 'safe']
    unsafe = df[df['label'] == 'unsafe']
    safe.to_csv(raw_dir / 'xstest_safe.csv', index=False)
    unsafe.to_csv(raw_dir / 'xstest_unsafe.csv', index=False)
    print(f'xstest: {len(safe)} safe and {len(unsafe)} unsafe prompts separated')
    return safe, unsafe


# Define function to append download records to the provenance file
def write_provenance(records, provenance_path):
    if not records:
        return
    file_exists = provenance_path.exists()
    with open(provenance_path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)
    print(f'provenance: {len(records)} records appended to {provenance_path}')


if __name__ == '__main__':
    download_records = download_all(sources=SOURCES, output_dir=RAW_DIR)
    xstest_safe, xstest_unsafe = split_xstest(raw_dir=RAW_DIR)
    write_provenance(records=download_records, provenance_path=PROVENANCE_PATH)
