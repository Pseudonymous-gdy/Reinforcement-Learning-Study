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


def init_worker():
    # ensure BLAS/OMP thread limits in worker processes
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


def worker_run(spec: RunSpec):
    try:
        # heartbeat: write periodic timestamp so the main process can observe liveness
        outdir_env = os.environ.get('FUSION_PILOT_OUTDIR', None)
        if outdir_env is None:
            hb_dir = os.path.join('/tmp', 'fusion_pilot_heartbeats')
        else:
            hb_dir = os.path.join(outdir_env, 'worker_heartbeats')
        try:
            os.makedirs(hb_dir, exist_ok=True)
        except Exception:
            hb_dir = '/tmp'

        stop_hb = threading.Event()

        def _hb_writer():
            name = f"{spec.case_name}__{spec.regime}__{spec.delta}__{spec.seed_id}__pid{os.getpid()}"
            path = os.path.join(hb_dir, name + '.heartbeat')
            while not stop_hb.is_set():
                try:
                    with open(path, 'w') as fh:
                        fh.write(str(time.time()))
                except Exception:
                    pass
                stop_hb.wait(2.0)

        hb_thread = threading.Thread(target=_hb_writer, daemon=True)
        hb_thread.start()
        base = spec.base_seed
        # deterministic instance generation (same across regimes)
        X, theta, i_star, meta = generate_instance(spec.case_name, spec.seed_id, base_seed=base)

        # create a deterministic run seed that depends on regime and delta
        run_seed = stable_int_seed(base, spec.case_name, spec.regime, spec.delta, spec.seed_id, 'run')

        regime_map = {'reward-only': 'reward_only', 'duel-only': 'duel_only', 'fusion': 'fusion'}
        alg_regime = regime_map.get(spec.regime, spec.regime)

        # call the algorithm (it internally uses the provided seed for RNG)
        res = run_phased_mixed_xy_bai(X, theta, spec.delta, alg_regime, run_seed, config=None)

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
        }
        return out
    except Exception as e:
            tb = traceback.format_exc()
            # include traceback in error_message for diagnostics
            return {
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
            }
    finally:
        try:
            stop_hb.set()
            hb_thread.join(timeout=1.0)
        except Exception:
            pass


def run(seeds=20, outdir='outputs/fusion_pilot_smoke', num_workers=None, chunksize=1, write_every=100, resume=False, overwrite=False, debug=False):
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

    # Use multiprocessing.Pool with spawn start method and initializer to
    # reduce risk of forking large memory and to set per-worker env vars.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # start method already set
        pass

    # export outdir for worker heartbeat files
    os.environ['FUSION_PILOT_OUTDIR'] = outdir

    pool = multiprocessing.Pool(processes=num_workers, initializer=init_worker, maxtasksperchild=50)
    try:
        # use imap_unordered to stream results as they complete
        for res in pool.imap_unordered(worker_run, run_specs, chunksize):
            # skip completed keys if res corresponds to already completed
            key = (res.get('case'), res.get('regime'), float(res.get('delta') or 0.0), int(res.get('seed_id') or 0))
            if key in completed_keys:
                pbar.update(1)
                continue
            if res.get('error', False):
                err_buffer.append(res)
            else:
                buffer.append(res)
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
    args = parser.parse_args()
    run(seeds=args.seeds, outdir=args.out, num_workers=args.num_workers, chunksize=args.chunksize, write_every=args.write_every, resume=args.resume, overwrite=args.overwrite, debug=args.debug)
