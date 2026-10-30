"""Verify the published snapshot, training coverage and executed source identity."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    entries = json.loads((ROOT / 'EVIDENCE-SHA256.json').read_text('utf-8'))
    for name, expected in entries.items():
        assert sha(ROOT / name) == expected, name
    base = ROOT / 'results/training/markdown-v3'
    records = [json.loads(line) for line in (base / 'metrics-complete.jsonl').read_text('utf-8').splitlines()]
    assert [r['phase_step'] for r in records] == list(range(1, 4797))
    assert records[-1]['phase_samples'] == 1841540
    proof = json.loads((base / 'verified-final.json').read_text('utf-8'))['verification']
    assert proof['phase_step'] == 4796 and proof['phase_samples'] == proof['total_phase_samples'] == 1841540
    assert proof['pvc_s3_sha256_match'] and proof['all_rank_preflight_passed']
    assert records[-1]['train_loss'] == proof['train_loss']
    config = json.loads((base / 'snapshot-20261020/config.json').read_text('utf-8'))
    for name, expected in config['code_sha256'].items():
        assert sha(ROOT / 'experiments/markdown-v3/code' / name) == expected, name
    corrected = json.loads((ROOT / 'results/training/corrective-v1/training-complete.json').read_text('utf-8'))
    assert corrected['updates'] == 38 and corrected['examples_consumed'] == 600
    assert corrected['automatically_promoted'] is False
    for name, expected in corrected['best_by_loss']['checkpoint']['phase_config']['code_sha256'].items():
        assert sha(ROOT / 'experiments/corrective-v1/code' / name) == expected, name
    print(f'{len(entries)} file hashes verified; 4796 contiguous updates; final proof and executed sources match.')
    print('Corrective run complete; candidate remains unpromoted. No network or GPU used.')

if __name__ == '__main__':
    main()
