"""Run only the authorized public benchmark images against the official OCR API.

No reference text is sent. The API credential is read in memory only; it is
never included in source, process arguments, results, headers saved to disk,
or console output. Existing successful requests are resumed without rebilling.
"""
import argparse
import base64
import getpass
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'unlimited-ocr2-comparison-20261028/livrables/comparatif-31-images-4-modeles'
API = 'https://api.mistral.ai/v1'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-stdin', action='store_true')
    parser.add_argument('--model', default='mistral-ocr-latest')
    args = parser.parse_args()
    if args.key_stdin:
        print('API_KEY_STDIN_READY', flush=True)
        key = (getpass.getpass('API key (hidden): ') if sys.stdin.isatty() else sys.stdin.readline()).strip()
    else:
        key = os.environ.get('MISTRAL_API_KEY') or getpass.getpass('Mistral API key (hidden): ')
    if not key or any(c.isspace() for c in key):
        raise RuntimeError('Missing or invalid API credential; no request sent')
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def call(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(API + path, data=data, headers={
            'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
            'User-Agent': 'Sovereign-Document-Suite-Public-OCR-Benchmark/1.0'})
        t = time.perf_counter()
        try:
            with opener.open(request, timeout=240) as response:
                raw = response.read()
                info = {'http_status': response.status, 'request_id': response.headers.get('x-request-id'),
                        'seconds': time.perf_counter() - t}
            return json.loads(raw), info
        except urllib.error.HTTPError as error:
            # Server error bodies are not printed: they could echo credentials.
            raise RuntimeError('Mistral HTTP ' + str(error.code)) from None
        except urllib.error.URLError as error:
            raise RuntimeError('Mistral network error: ' + type(error.reason).__name__) from None

    manifest = json.loads((SOURCE / 'jeu.json').read_text(encoding='utf-8'))
    assert manifest['counts'] == {'ar': 15, 'fr': 8, 'en': 8}
    assert manifest['dataset'] == 'Cognitive-Lab/NayanaOCRBench_Natural'
    results = ROOT / 'results/mistral'
    results.mkdir(parents=True, exist_ok=True)
    state_file = ROOT / 'execution.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'api': API + '/ocr',
        'requested_alias': args.model, 'sent_only_public_images': True, 'ground_truth_sent': False,
        'input_mode': 'same frozen 1024px RGB PNG bytes as Unlimited-OCR-2', 'expected': 31,
        'source_manifest_sha256': sha((SOURCE / 'jeu.json').read_bytes())}
    model = state.get('pinned_model') or args.model
    if not state.get('pinned_model'):
        models, _ = call('/models')
        available = [r for r in models.get('data', []) if 'ocr' in r.get('id', '').lower()]
        write(ROOT / 'metadata/available-ocr-models.json', {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'data': available})
        pinned = [r['id'] for r in available if args.model in r.get('aliases', []) and r['id'] != args.model]
        if len(pinned) == 1:
            model = pinned[0]
        state['pinned_model'] = model
        write(state_file, state)
        print('OCR_MODEL_SELECTED', model, flush=True)
    for index, case in enumerate(manifest['cases']):
        target = results / (case['id'] + '.json')
        raw_image = (SOURCE / case['inference_image']).read_bytes()
        digest = sha(raw_image)
        assert digest == case['inference_image_sha256']
        if target.exists():
            old = json.loads(target.read_text(encoding='utf-8'))
            assert old['image_sha256'] == digest and old['ground_truth_given_to_model'] is False
            print('RESUME', case['id'], 'already recorded', flush=True)
            continue
        payload = {'model': model, 'document': {'type': 'image_url',
                   'image_url': 'data:image/png;base64,' + base64.b64encode(raw_image).decode()},
                   'include_image_base64': False}
        # table_format omitted: default inline Markdown; headers/footer retained.
        response, http = call('/ocr', payload)
        pages = response.get('pages')
        assert isinstance(pages, list) and len(pages) == 1, 'Expected exactly one OCR page'
        assert isinstance(pages[0].get('markdown'), str), 'No OCR Markdown field'
        actual_model = response.get('model')
        if state.get('response_model'):
            assert actual_model == state['response_model'], 'API model changed during benchmark'
        else:
            state['response_model'] = actual_model
            # If discovery only provided an alias, pin the actual OCR response ID.
            if model.endswith('-latest') and actual_model and actual_model != model:
                model = actual_model
                state['pinned_model'] = model
        record = {'id': case['id'], 'image_sha256': digest, 'ground_truth_given_to_model': False,
                  'requested_model': payload['model'], 'model': actual_model, 'status': 'ok',
                  'text': pages[0]['markdown'], 'response': response, 'http': http,
                  'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                  'request_parameters_without_image': {'model': payload['model'], 'document_type': 'image_url',
                     'include_image_base64': False, 'table_format': 'omitted, default inline Markdown',
                     'extract_header': 'omitted, default false', 'extract_footer': 'omitted, default false'}}
        write(target, record)
        target.with_suffix('.md').write_text(record['text'], encoding='utf-8')
        state['completed'] = index + 1
        state['updated_at_utc'] = record['completed_at_utc']
        write(state_file, state)
        print('OCR_RECORDED', case['id'], f'{index+1}/31', actual_model, f'{http["seconds"]:.2f}s',
              'characters=' + str(len(record['text'])), flush=True)
    state['complete'] = True
    state['completed'] = 31
    state['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    write(state_file, state)
    print('ALL_31_REAL_MISTRAL_OUTPUTS_RECORDED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('BENCHMARK_STOPPED', type(error).__name__, str(error)[:180], flush=True)
        sys.exit(1)
