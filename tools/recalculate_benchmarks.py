"""Reproduce 248 archived paired scores, offline, using their original metrics."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def main():
    sys.path.insert(0, str(ROOT / 'evaluation/four_models'))
    four = module('four_visible', ROOT / 'evaluation/four_models/mesures_visibles.py')
    common = module('paired_metrics', ROOT / 'evaluation/printed/metrics.py')
    total = 0
    for key in ('printed-four-models', 'printed-mistral-unlimited', 'handwritten-mistral-unlimited'):
        d = json.loads((ROOT / 'results/benchmarks' / key / 'resultats.json').read_text(encoding='utf-8'))
        assert len(d['cases']) == 31
        counts = {'ar': 0, 'fr': 0, 'en': 0}
        sums = {}
        for case in d['cases']:
            counts[case['language']] += 1
            for model, out in case['outputs'].items():
                reference, prediction = case['reference'], out['text']
                if key == 'handwritten-mistral-unlimited' and case['language'] == 'en':
                    reference, prediction = reference.replace('^', ''), prediction.replace('^', '')
                score = four.score(reference, prediction) if key == 'printed-four-models' else common.score(reference, prediction)
                assert score == out['quality'], (key, case['id'], model)
                for lang in ('all', case['language']):
                    row = sums.setdefault((lang, model), [0, 0, 0, 0])
                    for i, field in enumerate(('character_errors', 'reference_characters', 'word_errors', 'reference_words')):
                        row[i] += score[field]
                total += 1
        assert counts == {'ar': 15, 'fr': 8, 'en': 8}
        for (lang, model), (ce, nc, we, nw) in sums.items():
            expected = d['aggregate'][lang][model]
            assert abs(100 * ce / nc - expected['cer']) < 1e-10, (key, lang, model, 'cer')
            assert abs(100 * we / nw - expected['wer']) < 1e-10, (key, lang, model, 'wer')
        print(key + ': all archived per-image and aggregate scores match')
    assert total == 248
    print('248 archived scores reproduced. No inference, network, key or GPU used.')

if __name__ == '__main__':
    main()
