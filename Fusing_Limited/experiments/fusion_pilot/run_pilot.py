import os
# limit BLAS/OMP threads to avoid oversubscription in worker processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import argparse
import hashlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import traceback
import threading
import time
import json
import queue
from .instances import make_unit_instance, make_special_instance, make_general_instance
from .algorithm import run_phased_mixed_xy_bai
from .config import delta_values, omega_R_values


@dataclass(frozen=True)
class RunSpec:
    case_name: str
    regime: str
    delta: float
    seed_id: int
    base_seed: int = 20260601


def stable_int_seed(*items, modulo=2 ** 32 - 1):
    text = "::".join(map(str, items))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


_PROGRESS_QUEUE = None

def init_worker(progress_queue):
    # set BLAS/OMP limits and register the progress queue for workers
    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = progress_queue
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'


def make_rng(base_seed, *items):
    s = stable_int_seed(base_seed, *items)
    return np.random.default_rng(s)


def generate_instance(case_name, seed_id, base_seed=20260601):
    if case_name == 'unit':
        X, theta, i_star, meta = make_unit_instance()
        return X, theta, i_star, meta
    if case_name == 'special':
        X, theta, i_star, meta = make_special_instance()
        return X, theta, i_star, meta
    # general: use instance RNG independent of regime/delta
    instance_rng = make_rng(base_seed, case_name, 'instance', seed_id)
    X, theta, i_star, meta = make_general_instance(seed=seed_id, rng=instance_rng)
    return X, theta, i_star, meta


def worker_run(spec: RunSpec, design_solver=None, debug_cap_n_m=None, verbose=False):
    try:
        # emit run_start event
        start_ts = time.time()
        if _PROGRESS_QUEUE is not None:
            try:
                _PROGRESS_QUEUE.put_nowait({
                    'timestamp': start_ts,
                    'event': 'run_start',
                    'case': spec.case_name,
                    'regime': spec.regime,
                    'delta': spec.delta,
                    'seed_id': spec.seed_id,
                    'worker_pid': os.getpid(),
                    'elapsed_sec': 0.0,
                })
            except queue.Full:
                pass
        base = spec.base_seed
        # deterministic instance generation (same across regimes)
        X, theta, i_star, meta = generate_instance(spec.case_name, spec.seed_id, base_seed=base)

        # create a deterministic run seed that depends on regime and delta
        run_seed = stable_int_seed(base, spec.case_name, spec.regime, spec.delta, spec.seed_id, 'run')

        alg_regime = spec.regime

        # prepare progress_emitter for algorithm phases
        def _emit(ev):
            if _PROGRESS_QUEUE is None:
                return
            try:
                # fill in identifying info
                ev['case'] = spec.case_name
                ev['regime'] = spec.regime
                ev['delta'] = spec.delta
                ev['seed_id'] = spec.seed_id
                ev['worker_pid'] = os.getpid()
                ev['timestamp'] = time.time()
                _PROGRESS_QUEUE.put_nowait(ev)
            except Exception:
                pass

        # call the algorithm (it internally uses the provided seed for RNG)
        config = {
            'design_solver': design_solver,
            'debug_cap_n_m': debug_cap_n_m,
            'verbose': verbose,
        }
        res = run_phased_mixed_xy_bai(X, theta, spec.delta, alg_regime, run_seed, config=config, progress_emitter=_emit)

        out = {
            'case': spec.case_name,
            'regime': spec.regime,
            'delta': spec.delta,
            'seed_id': spec.seed_id,
            'success': bool(res.get('success', False)),
            'error': False,
            'error_type': '',
            'error_message': '',
            'best_arm_true': int(res.get('i_star', i_star)),
            'best_arm_hat': int(res.get('i_hat', -1)),
            'T_r': int(res.get('T_r', 0)),
            'T_d': int(res.get('T_d', 0)),
            'T_total': int(res.get('T_total', 0)),
            'num_phases': int(res.get('final_phase', 0)),
            'T_burn_r': int(res.get('T_r_burn', 0)),
            'T_burn_d': int(res.get('T_d_burn', 0)),
            'T_main_r': int(res.get('T_r_main', 0)),
            'T_main_d': int(res.get('T_d_main', 0)),
            'p_D': float(res.get('p_D', 0.0)),
            # final_gap_hat: true gap between best and second-best under true theta
            # compute scores on true theta and report best-minus-second gap
            'final_gap_hat': float(np.sort(X @ theta)[-1] - np.sort(X @ theta)[-2]) if X.shape[0] > 1 else 0.0,
            'stop_stat': float(res.get('final_loglik', 0.0)),
            'elapsed_sec': time.time() - start_ts,
        }
        return out
    except Exception as e:
        tb = traceback.format_exc()
        # include traceback in error_message for diagnostics
        err_out = {
            'case': spec.case_name,
            'regime': spec.regime,
            'delta': spec.delta,
            'seed_id': spec.seed_id,
            'success': False,
            'error': True,
            'error_type': type(e).__name__,
            'error_message': f"{str(e)}\n{tb}",
            'best_arm_true': None,
            'best_arm_hat': None,
            'T_r': 0,
            'T_d': 0,
            'T_total': 0,
            'num_phases': 0,
            'T_burn_r': 0,
            'T_burn_d': 0,
            'T_main_r': 0,
            'T_main_d': 0,
            'p_D': 0.0,
            'final_gap_hat': 0.0,
            'stop_stat': 0.0,
            'elapsed_sec': time.time() - start_ts,
        }
        # emit run_error
        if _PROGRESS_QUEUE is not None:
            try:
                _PROGRESS_QUEUE.put_nowait({
                    'timestamp': time.time(),
                    'event': 'run_error',
                    'case': spec.case_name,
                    'regime': spec.regime,
                    'delta': spec.delta,
                    'seed_id': spec.seed_id,
                    'worker_pid': os.getpid(),
                    'error_type': err_out['error_type'],
                    'error_message': err_out['error_message'],
                    'elapsed_sec': err_out['elapsed_sec'],
                })
            except queue.Full:
                pass
        return err_out


def worker_run_star(args):
    return worker_run(*args)


def run(seeds=20, outdir='outputs/fusion_pilot_smoke', num_workers=None, chunksize=1, write_every=100, resume=False, overwrite=False, debug=False, design_solver='greedy-fw', debug_cap_n_m=None):
    ensure_out(outdir)
    cases = ['unit', 'special', 'general']
    regimes = ['reward-only', 'duel-only', 'fusion']
    deltas = delta_values

    # build run specs
    if debug:
        seeds_list = list(range(min(2, seeds)))
        deltas = [0.05]
    else:
        seeds_list = list(range(seeds))

    run_specs = []
    for case in cases:
        for regime in regimes:
            for delta in deltas:
                for seed_id in seeds_list:
                    run_specs.append(RunSpec(case, regime, delta, seed_id))

    assert len(run_specs) == (3 * 3 * len(deltas) * len(seeds_list))

    out_raw = os.path.join(outdir, 'raw_results.csv')
    out_err = os.path.join(outdir, 'error_results.csv')

    completed_keys = set()
    if resume and os.path.exists(out_raw):
        existing = pd.read_csv(out_raw)
        for _, r in existing.iterrows():
            if not r.get('error', False):
                completed_keys.add((r['case'], r['regime'], float(r['delta']), int(r['seed_id'])))
    if overwrite and os.path.exists(out_raw):
        os.remove(out_raw)
    # prepare executor
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    total = len(run_specs)
    pbar = tqdm(total=total, desc='Total runs')
    buffer = []
    err_buffer = []
    write_header = not os.path.exists(out_raw) or overwrite

    def flush_buffer(buf, path, header):
        if len(buf) == 0:
            return
        df = pd.DataFrame(buf)
        df.to_csv(path, mode='a', header=header, index=False)

    # Use multiprocessing.Pool with chosen start method and initializer to
    # centralize progress reporting.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    mp_ctx = multiprocessing.get_context('spawn')
    progress_queue = mp_ctx.Queue(maxsize=10000)

    # start progress writer thread in main process
    progress_events_path = os.path.join(outdir, 'progress_events.jsonl')
    progress_snapshot_path = os.path.join(outdir, 'progress_snapshot.json')
    prog_stop = threading.Event()

    def progress_writer_loop(queue_obj, events_path, snapshot_path, snapshot_every=10, print_every=30):
        # maintain simple stats
        stats = {
            'total_runs': len(run_specs),
            'submitted_runs': 0,
            'completed_runs': 0,
            'error_runs': 0,
            'running_runs': 0,
            'by_key': {},
        }
        last_snapshot = time.time()
        last_print = time.time()
        # open file for append
        with open(events_path, 'a') as ef:
            while not prog_stop.is_set():
                try:
                    ev = queue_obj.get(timeout=1.0)
                except Exception:
                    ev = None
                if ev is not None:
                    ef.write(json.dumps(ev, default=str) + '\n')
                    ef.flush()
                    # update basic stats
                    key = (ev.get('case'), ev.get('regime'), float(ev.get('delta') or 0.0))
                    if ev.get('event') == 'run_start':
                        stats['running_runs'] += 1
                        stats['submitted_runs'] += 1
                        stats['by_key'].setdefault(key, {'completed': 0, 'errors': 0, 'running': 0})
                        stats['by_key'][key]['running'] += 1
                    elif ev.get('event') == 'run_end':
                        stats['running_runs'] = max(0, stats['running_runs'] - 1)
                        stats['completed_runs'] += 1
                        stats['by_key'].setdefault(key, {'completed': 0, 'errors': 0, 'running': 0})
                        stats['by_key'][key]['running'] = max(0, stats['by_key'][key]['running'] - 1)
                        stats['by_key'][key]['completed'] += 1
                    elif ev.get('event') == 'run_error':
                        stats['running_runs'] = max(0, stats['running_runs'] - 1)
                        stats['error_runs'] += 1
                        stats['by_key'].setdefault(key, {'completed': 0, 'errors': 0, 'running': 0})
                        stats['by_key'][key]['running'] = max(0, stats['by_key'][key]['running'] - 1)
                        stats['by_key'][key]['errors'] += 1
                now = time.time()
                if now - last_snapshot >= snapshot_every:
                    # write snapshot atomically
                    snap = {
                        'total_runs': stats['total_runs'],
                        'submitted_runs': stats['submitted_runs'],
                        'completed_runs': stats['completed_runs'],
                        'error_runs': stats['error_runs'],
                        'running_runs': stats['running_runs'],
                        'completion_percent': (stats['completed_runs'] / max(1, stats['total_runs'])) * 100.0,
                    }
                    tmp = snapshot_path + '.tmp'
                    try:
                        with open(tmp, 'w') as sf:
                            json.dump(snap, sf)
                        os.replace(tmp, snapshot_path)
                    except Exception:
                        pass
                    last_snapshot = now
                if now - last_print >= print_every:
                    print(f"completed={stats['completed_runs']}/{stats['total_runs']} errors={stats['error_runs']} running={stats['running_runs']}")
                    last_print = now

    prog_thread = threading.Thread(target=progress_writer_loop, args=(progress_queue, progress_events_path, progress_snapshot_path, 10, 30), daemon=True)
    prog_thread.start()

    pool = mp_ctx.Pool(processes=num_workers, initializer=init_worker, initargs=(progress_queue,), maxtasksperchild=50)
    try:
        # use imap_unordered to stream results as they complete
        # filter completed if resume requested
        run_specs_to_execute = []
        if resume and os.path.exists(out_raw):
            existing = pd.read_csv(out_raw)
            for _, r in existing.iterrows():
                if not r.get('error', False):
                    completed_keys.add((r['case'], r['regime'], float(r['delta']), int(r['seed_id'])))
        for spec in run_specs:
            key = (spec.case_name, spec.regime, spec.delta, spec.seed_id)
            if key in completed_keys:
                pbar.update(1)
                continue
            run_specs_to_execute.append(spec)

        worker_args = [(spec, design_solver, debug_cap_n_m, debug) for spec in run_specs_to_execute]
        for res in pool.imap_unordered(worker_run_star, worker_args, chunksize):
            # skip completed keys if res corresponds to already completed
            key = (res.get('case'), res.get('regime'), float(res.get('delta') or 0.0), int(res.get('seed_id') or 0))
            if key in completed_keys:
                pbar.update(1)
                continue
            if res.get('error', False):
                err_buffer.append(res)
                # also emit run_end or run_error events
                if _PROGRESS_QUEUE is not None:
                    try:
                        ev = {'timestamp': time.time(), 'event': 'run_end' if not res.get('error') else 'run_error', 'case': res.get('case'), 'regime': res.get('regime'), 'delta': res.get('delta'), 'seed_id': res.get('seed_id'), 'worker_pid': os.getpid(), 'elapsed_sec': res.get('elapsed_sec', 0.0)}
                        progress_queue.put_nowait(ev)
                    except Exception:
                        pass
            else:
                buffer.append(res)
                if _PROGRESS_QUEUE is not None:
                    try:
                        ev = {'timestamp': time.time(), 'event': 'run_end', 'case': res.get('case'), 'regime': res.get('regime'), 'delta': res.get('delta'), 'seed_id': res.get('seed_id'), 'worker_pid': os.getpid(), 'elapsed_sec': res.get('elapsed_sec', 0.0)}
                        progress_queue.put_nowait(ev)
                    except Exception:
                        pass
            if len(buffer) >= write_every:
                flush_buffer(buffer, out_raw, write_header)
                write_header = False
                buffer.clear()
            if len(err_buffer) >= write_every:
                flush_buffer(err_buffer, out_err, True)
                err_buffer.clear()
            pbar.update(1)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise
    except Exception as e:
        # catastrophic worker failure (segfault/kill) will surface here; record and continue
        tb = traceback.format_exc()
        err = {'case': 'pool-level', 'regime': '', 'delta': 0.0, 'seed_id': -1, 'success': False,
               'error': True, 'error_type': type(e).__name__, 'error_message': f"{str(e)}\n{tb}"}
        err_buffer.append(err)
    finally:
        pool.close()
        pool.join()

    # flush remaining
    flush_buffer(buffer, out_raw, write_header)
    flush_buffer(err_buffer, out_err, True)
    pbar.close()
    # save config
    cfg = dict(delta_values=delta_values, omega_R_values=omega_R_values)
    with open(os.path.join(outdir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Saved raw results to', out_raw)


def ensure_out(outdir):
    os.makedirs(outdir, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--out', type=str, default='outputs/fusion_pilot_smoke')
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--chunksize', type=int, default=1)
    parser.add_argument('--write-every', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--design-solver', type=str, default='greedy-fw', choices=['greedy-fw', 'slsqp'])
    parser.add_argument('--debug-cap-nm', type=int, default=None)
    args = parser.parse_args()
    run(seeds=args.seeds, outdir=args.out, num_workers=args.num_workers, chunksize=args.chunksize, write_every=args.write_every, resume=args.resume, overwrite=args.overwrite, debug=args.debug, design_solver=args.design_solver, debug_cap_n_m=args.debug_cap_nm)
