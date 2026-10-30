"""Verified fast-kernel full pass, isolated from every previous training run.

Starts from the best pilot adapter/optimizer, but resets the full-data cursor.
Pilot samples were not a contiguous prefix of the full stream. No file pruning.
"""
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Literal

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['TRAIN_MODE'] = 'train'

import numpy as np
import torch
import torch.distributed as dist
import arsenal_ocr
from arsenal_ocr import OCRModel, AdamConfig, ROOT as SOURCE, BUCKET
from distributed_ocr import RankStream, DistributedTraining
from pilot_train import PilotTrainer
from ordered_prefetch import OrderedPrefetch
from minio_retry import read_object, object_sha256
from train_full import connect
from stage_training import DATA_DIR
from arsenal.config.base import BaseConfig
from arsenal.config.dataloader import BaseDataloaderConfig
from arsenal.config.distributed import DistributedConfig
from arsenal.config.hsdp import HsdpConfig
from arsenal.config.compile import CompileConfig
from lr_scheduler_basic import CosineSchedulerConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

RUN = SOURCE / 'training-fast-v1'
TOTAL = 3683080
BATCH = 255  # 85 examples per rank with three GPUs; preserves equal DDP work.
RANKS = int(os.environ.get('TRAIN_RANKS', '4'))
PARENT_STEP = None
PARENT_SHA = None
assert 1 <= RANKS <= 4
SOURCE_SHA = 'd9a45e38c05720708aaf24282f93050920a158aae8987aa449395c04d9597117'
SOURCE_KEY = 'pilot-10h-v1/checkpoint-3205.pt'
CALLS = {}
assert qwen.is_fast_path_available, 'Fast kernels missing: do not fall back silently'
import triton
assert triton.__version__ == '3.7.1'
for name in ['causal_conv1d_fn', 'chunk_gated_delta_rule']:
    original = getattr(qwen, name)
    def counted(*args, _fn=original, _name=name, **kwargs):
        CALLS[_name] = CALLS.get(_name, 0) + 1
        return _fn(*args, **kwargs)
    setattr(qwen, name, counted)


def write(path, data):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(data, indent=2, default=str))
    temp.replace(path)


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def verify_s3(key, sha):
    actual = object_sha256(connect(streaming=True), BUCKET, key)
    assert actual == sha, 'S3 checksum mismatch: ' + key


def upload(path):
    key = 'training-fast-v1/' + str(path.relative_to(RUN))
    sha = digest(path)
    connect().upload_file(str(path), BUCKET, key)
    verify_s3(key, sha)
    return {'path': str(path), 'key': key, 'sha256': sha}


class FastStream(RankStream):
    def load_state_dict(self, state):
        # Only initial pilot import may reset to the first full-data example.
        if state.get('pilot'):
            state = {'shard': 0, 'cursor': 0, 'count': 0}
        super().load_state_dict(state)

    def __iter__(self):
        for batch in super().__iter__():
            if os.environ.get('Q35OCR_PIN_EACH_SAMPLE', '1') == '1':
                yield type(batch)({k: v.pin_memory() for k, v in batch.items()})
            else:
                yield batch


class FastData(BaseDataloaderConfig):
    kind: Literal['q35ocr_fast_full'] = 'q35ocr_fast_full'
    model_path: str

    def spawn(self):
        return OrderedPrefetch(FastStream(self.model_path), depth=4)


class FastConfig(BaseConfig):
    model: OCRModel


class FastTrainer(PilotTrainer):
    def setup(self):
        arsenal_ocr.ROOT = RUN
        super().setup()

    def lr_scheduler_step(self):
        self.lr_scheduler.set_lr(step=min(460385, 1 + (self.source_samples + self.consumed) // 8))

    def close_loader(self):
        if getattr(self, 'iterator', None) is not None:
            self.iterator.close()
            self.iterator = None
        self.dataloader.close()

    def batch_get(self, iterator):
        raw = next(iterator)
        if raw is None:
            self._dummy_batch = True
            return None
        self._dummy_batch = False
        return super().batch_get(iter([raw]))

    def compute_loss(self, batch):
        if batch is None:
            # All ranks must enter the DDP reduction for the final remainder,
            # while ranks without a real sample contribute exactly zero.
            loss=sum((parameter.float().sum() * 0.0) for parameter in self.get_parameters())
            self.record_window_loss(loss)
            return loss
        return super().compute_loss(batch)

    def restore_full(self, entry):
        self.close_loader()
        path = Path(entry['path'])
        assert digest(path) == entry['sha256']
        if dist.get_rank() == 0:
            verify_s3(entry['key'], entry['sha256'])
        dist.barrier()
        DistributedTraining.restore(self, path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        saved_optimization = state.get('optimization_settings')
        if saved_optimization is not None:
            requested_optimization = {
                'global_batch': BATCH,
                'microbatch': int(os.environ.get('MICROBATCH', '1')),
                'accumulation': int(os.environ.get('Q35OCR_CONFIGURED_ACCUMULATION', self.config.gradient_accumulation.count)),
                'lora_compute_dtype': os.environ.get('Q35OCR_LORA_COMPUTE_DTYPE', 'fp32'),
                'rank_sync_interval': int(os.environ.get('Q35OCR_RANK_SYNC_INTERVAL', '16')),
            }
            parent_migration = PARENT_STEP is not None and (
                entry.get('global_step') == PARENT_STEP and entry.get('sha256') == PARENT_SHA
            )
            assert parent_migration or saved_optimization == requested_optimization, (
                'Refusing exact resume with changed optimization settings',
                saved_optimization,
                requested_optimization,
            )
        self.consumed = state['full_samples']
        self.source_samples = state['source_samples']
        assert self.dataloader.count == self.consumed
        self.iterator = iter(self.dataloader)
        self.sync_check()

    def save_full(self, label, publish=True):
        states = [None] * RANKS
        dist.all_gather_object(states, self.dataloader.state_dict())
        canonical = max(states, key=lambda state: state['count'])
        assert canonical['count'] == self.consumed
        counts = sorted(x['count'] for x in states)
        regular = list(range(self.consumed - RANKS + 1, self.consumed + 1))
        remainder = [self.consumed - 1] * (RANKS - 1) + [self.consumed]
        assert counts == regular or counts == sorted(remainder) or self.consumed == 0
        local = {'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}
        rng = [None] * RANKS
        dist.all_gather_object(rng, local)
        result = [None]
        if dist.get_rank() == 0:
            path = RUN / ('checkpoint-' + label + '.pt')
            assert not path.exists(), 'Refusing to overwrite a retained checkpoint'
            state = {'adapter': self.adapter_state(), 'optimizer': self.optimizer.state_dict(),
                     'clock': self.clock.state_dict(), 'loader': canonical, 'rank_rng': rng,
                      'world_size': RANKS, 'scheduler_step': min(460385, 1 + (self.source_samples + self.consumed) // 8),
                     'source_samples': self.source_samples, 'full_samples': self.consumed,
                     'source_checkpoint_sha256': SOURCE_SHA, 'config': self.config.model_dump_json(),
                 'optimization_settings': {
                     'global_batch': BATCH,
                     'microbatch': int(os.environ.get('MICROBATCH', '1')),
                     'accumulation': int(os.environ.get('Q35OCR_CONFIGURED_ACCUMULATION', self.config.gradient_accumulation.count)),
                     'lora_compute_dtype': os.environ.get('Q35OCR_LORA_COMPUTE_DTYPE', 'fp32'),
                     'rank_sync_interval': int(os.environ.get('Q35OCR_RANK_SYNC_INTERVAL', '16')),
                 },
                     'data_complete_sha256': digest(DATA_DIR / 'COMPLETE.json')}
            temp = path.with_suffix('.partial')
            torch.save(state, temp)
            temp.replace(path)
            result[0] = upload(path)
            result[0].update(global_step=self.clock.step_index, full_samples=self.consumed)
            if publish:
                write(RUN / 'latest.json', result[0])
                upload(RUN / 'latest.json')
            print('FAST_CHECKPOINT_VERIFIED', json.dumps(result[0]), flush=True)
        dist.broadcast_object_list(result, src=0)
        return result[0]

    def evaluate(self):
        # Use exactly the pilot validation set and token-weighted objective.
        # Its 256 examples are partitioned over the active ranks to avoid idle GPUs.
        self.close_loader()
        state = self.dataloader.state_dict()
        self.dataloader.load_state_dict(state)
        model = self.models['model'].module
        model.eval()
        total = 0.
        tokens = 0
        validation_path=Path(os.environ.get('Q35OCR_VALIDATION_FILE',str(SOURCE/'pilot-10h-v1/validation.jsonl')))
        rows = [json.loads(line) for line in validation_path.read_text().splitlines()]
        assert len(rows) == 256
        s3 = connect(streaming=True)
        with torch.no_grad():
            for row in rows[dist.get_rank()::RANKS]:
                im = row['image']
                off = im['offset']
                raw = read_object(s3, BUCKET, im['key'],
                                  byte_range=f"bytes={off}-{off+im['size']-1}")
                batch = self.dataloader.inner.encode(row, raw).to(self.device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(batch)
                assert torch.isfinite(loss), 'Nonfinite validation loss'
                n = int((batch['labels'][:, 1:] != -100).sum())
                total += float(loss) * n
                tokens += n
        values = torch.tensor([total, tokens], device=self.device, dtype=torch.float64)
        dist.all_reduce(values)
        model.train()
        self.iterator = iter(self.dataloader)
        return float(values[0] / values[1])

    def update(self, count=BATCH):
        partial = count % RANKS != 0
        assert not partial or (count == TOTAL - self.consumed and count < RANKS)
        self.config.gradient_accumulation.count = count // RANKS if not partial else 1
        self.loss_sum = 0.
        self.loss_count = 0
        if partial:
            def final_iterator():
                try:
                    yield next(self.iterator)
                except StopIteration:
                    yield None
            self.step(final_iterator())
        else:
            self.step(self.iterator)
        self.clock.step_index += 1
        self.metrics.clear()
        self.consumed += count
        return self.mean_accumulated_loss()

    def preflight(self):
        source = SOURCE / SOURCE_KEY
        assert digest(source) == SOURCE_SHA
        if dist.get_rank() == 0:
            verify_s3(SOURCE_KEY, SOURCE_SHA)
        dist.barrier()
        DistributedTraining.restore(self, source)
        state = torch.load(source, map_location='cpu', weights_only=False)
        self.source_samples = state['source_samples'] + state['pilot_samples']
        self.consumed = 0
        self.dataloader.load_state_dict({'shard': 0, 'cursor': 0, 'count': 0})
        self.iterator = iter(self.dataloader)
        baseline = self.evaluate()
        assert math.isfinite(baseline)
        initial = self.save_full('initial', publish=False)
        original = self.adapter_state()
        self.update()
        assert any(not torch.equal(original[k], v) for k, v in self.adapter_state().items())
        del original
        checkpoint = self.save_full('preflight', publish=False)
        expected_loss = self.update()
        expected = self.adapter_state()
        self.restore_full(checkpoint)
        repeated_loss = self.update()
        error = max(float((expected[k] - v).abs().max()) for k, v in self.adapter_state().items())
        del expected
        assert error < 1e-5 and abs(expected_loss - repeated_loss) < 1e-4, (error, expected_loss, repeated_loss)
        self.sync_check()
        assert CALLS.get('causal_conv1d_fn', 0) and CALLS.get('chunk_gated_delta_rule', 0)
        proofs = [None] * RANKS
        dist.all_gather_object(proofs, {'rank': dist.get_rank(), 'resume_error': error, 'loss_error': abs(expected_loss-repeated_loss), 'kernel_calls': CALLS.copy(), 'gpu': torch.cuda.get_device_name()})
        self.restore_full(initial)
        if dist.get_rank() == 0:
            result = {'status': 'passed', 'baseline_eval_loss': baseline, 'ranks': proofs,
                      'gates': ['full_stream_cursor', 'finite_gradients', 'parameter_update', 'exact_resume', 'pvc_s3_verified_checkpoint', 'kernel_execution'],
                      'coverage': 'Preflight is rolled back. Start a new complete deterministic pass; pilot visits are not counted as full-pass coverage.'}
            write(RUN / 'preflight.json', result)
            upload(RUN / 'preflight.json')
            write(RUN / 'best.json', dict(initial, eval_loss=baseline))
            upload(RUN / 'best.json')
            print('FAST_PREFLIGHT_PASSED', json.dumps(result), flush=True)
        dist.barrier()
        return baseline

    def _train(self):
        try:
            if os.environ.get('Q35OCR_PREFLIGHT_ONLY', '0') == '1':
                preflight = getattr(self, 'preflight_resume', self.preflight)
                preflight()
                if dist.get_rank() == 0:
                    write(RUN / 'result.json', {
                        'state': 'preflight_passed_only',
                        'full_training_started': False,
                        'world_size': RANKS,
                        'global_batch': BATCH,
                        'microbatch': int(os.environ.get('MICROBATCH', '1')),
                    })
                    upload(RUN / 'result.json')
                dist.barrier()
                self.close_loader()
                self.clock.stop()
                return
            self.run_full()
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            write(RUN / f'failure-rank{dist.get_rank()}.json', {'time': time.time(), 'error': repr(exc), 'full_samples': getattr(self, 'consumed', None)})
            os._exit(1)

    def run_full(self):
        self.iterator = None
        best = self.preflight()
        started = time.time()
        last_save = started
        history = []
        bad = 0
        reason = 'complete'
        updates = 0
        while self.consumed < TOTAL:
            began = time.time()
            loss = self.update(min(BATCH, TOTAL - self.consumed))
            updates += 1
            history.append(loss)
            entry = {'timestamp': time.time(), 'state': 'running', 'global_step': self.clock.step_index,
                     'full_updates': updates, 'full_samples': self.consumed, 'total_samples': TOTAL,
                     'train_loss': loss, 'learning_rate': self.optimizer.param_groups[0]['lr'],
                     'update_seconds': time.time() - began, 'world_size': RANKS, 'global_batch': BATCH,
                     'microbatch': 1, 'accumulation': self.config.gradient_accumulation.count}
            stop = None
            if len(history) >= 13 and all(x > 3*statistics.median(history[-13:-3]) for x in history[-3:]):
                stop = 'divergent_loss'
            if len(history) >= 20 and statistics.pstdev(history[-10:]) > statistics.mean(history[-10:]) and statistics.mean(history[-10:]) > 1.5*statistics.median(history[-20:-10]):
                stop = 'unstable_loss'
            stop = self.decision(stop or ('requested_stop' if (RUN / 'STOP').exists() else None))
            low_disk = self.decision(shutil.disk_usage(RUN).free < 4*1024**3)
            if low_disk:
                stop = 'disk_reserve_reached_no_files_deleted'
            save = self.decision(updates == 1 or updates % 250 == 0 or time.time()-last_save >= 3600 or bool(stop) or self.consumed == TOTAL)
            if save:
                score = self.evaluate()
                entry['eval_loss'] = score
                self.sync_check()
                marker = self.save_full(str(self.clock.step_index))
                last_save = time.time()
                if score < best:
                    best = score
                    if dist.get_rank() == 0:
                        write(RUN / 'best.json', dict(marker, eval_loss=best))
                        upload(RUN / 'best.json')
                bad = bad+1 if score > best*1.2 and score-best > .05 else 0
                if bad >= 2:
                    stop = 'validation_degradation'
            elapsed = time.time()-started
            entry.update(elapsed_seconds=elapsed, examples_per_second=self.consumed/elapsed,
                         estimated_remaining_seconds=(TOTAL-self.consumed)*elapsed/self.consumed,
                         state='stopped' if stop else 'running', stop_reason=stop)
            if dist.get_rank() == 0:
                with (RUN / 'metrics.jsonl').open('a') as file:
                    file.write(json.dumps(entry)+'\n')
                write(RUN / 'status.json', entry)
                if save or updates % 10 == 0:
                    upload(RUN / 'metrics.jsonl')
                    upload(RUN / 'status.json')
                print('FAST_TRAIN_METRIC', json.dumps(entry), flush=True)
            if stop:
                reason = stop
                break
        self.close_loader()
        if dist.get_rank() == 0:
            result = {'state': 'complete' if self.consumed == TOTAL else 'stopped', 'reason': reason,
                      'full_samples': self.consumed, 'updates': updates, 'global_step': self.clock.step_index,
                      'elapsed_seconds': time.time()-started, 'best_eval_loss': best}
            write(RUN / 'result.json', result)
            upload(RUN / 'result.json')
        dist.barrier()
        self.clock.stop()


def main():
    assert not RUN.exists(), 'Inspect existing run; no automatic overwrite or restart'
    for mode in ['fast', 'baseline', 'prefetch']:
        key = f'optimization/benchmark-{mode}-v3/result.json'
        result = json.loads(connect().get_object(Bucket=BUCKET, Key=key)['Body'].read())
        assert result['status'] == 'passed' and len(result['ranks']) == RANKS
        assert result['source_checkpoint_sha256'] == SOURCE_SHA
        assert all(rank['resume_error'] < 1e-5 for rank in result['ranks'])
        if mode != 'baseline':
            assert all(rank['kernel_calls'].get('chunk_gated_delta_rule', 0) > 0 for rank in result['ranks'])
    assert shutil.disk_usage(SOURCE).free > 10*1024**3
    RUN.mkdir()
    (RUN / 'logs').mkdir()
    write(RUN / 'started.json', {'time': time.time(), 'source_checkpoint': SOURCE_KEY, 'source_sha256': SOURCE_SHA,
                                'coverage': 'One new complete pass of 3683080 examples, initialized from pilot3205',
                                'world_size': RANKS, 'global_batch': BATCH, 'microbatch': 1, 'accumulation': BATCH // RANKS,
                                'lora_compute_dtype': os.environ.get('Q35OCR_LORA_COMPUTE_DTYPE','fp32'),
                                'rank_sync_interval': int(os.environ.get('Q35OCR_RANK_SYNC_INTERVAL','16')),
                                'kernels': {'fla-core': '0.5.2', 'causal-conv1d': '1.7.0', 'triton': triton.__version__}})
    upload(RUN / 'started.json')
    config = FastConfig(run_directory=RUN/'arsenal', model=OCRModel(model_path=str(SOURCE/'model')),
                        data=FastData(model_path=str(SOURCE/'model'), batch_size=1), optimizer=AdamConfig(),
                        lr_scheduler=CosineSchedulerConfig(init_lr=1e-6, max_lr=1e-4, min_lr_ratio=.1, max_steps=460385, use_warmup=True, warmup_steps=500, alpha=1.),
                        distributed=DistributedConfig(device_type='cuda', num_process_per_node=RANKS, hsdp=HsdpConfig(replicate_size=RANKS, shard_size=1), use_local_loss=True),
                        compile=CompileConfig(enabled=False), dtype=torch.bfloat16)
    FastTrainer(config).train()


if __name__ == '__main__':
    main()
