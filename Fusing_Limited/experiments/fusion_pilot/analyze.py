import os
import pandas as pd
import numpy as np


def produce_cost_tables(raw_csv, outdir):
    df = pd.read_csv(raw_csv)
    # normalize column names
    if 'case' in df.columns and 'case_name' not in df.columns:
        df = df.rename(columns={'case': 'case_name'})
    if 'seed_id' in df.columns and 'seed' not in df.columns:
        df = df.rename(columns={'seed_id': 'seed'})
    os.makedirs(outdir, exist_ok=True)
    # cost_by_omega
    rows = []
    for _, r in df.iterrows():
        for omega in [1, 2, 4, 8, 16]:
            cost = omega * r['T_r'] + 1 * r['T_d']
            rows.append(dict(case_name=r['case_name'], regime=r['regime'], delta=r['delta'], seed=r['seed'], omega_R=omega, omega_D=1, cost=cost, T_r=r['T_r'], T_d=r['T_d'], T_total=r['T_total'], success=r.get('success', False), stopped=r.get('stopped', True)))
    cdf = pd.DataFrame(rows)
    cdf.to_csv(os.path.join(outdir, 'cost_by_omega.csv'), index=False)

    # fusion cost ratio
    fr = []
    # group by case, delta, seed and omega
    cases = df['case_name'].unique()
    deltas = df['delta'].unique()
    for case in cases:
        for delta in deltas:
            sub = df[(df['case_name'] == case) & (df['delta'] == delta)]
            seeds = sub['seed'].unique()
            for seed in seeds:
                rows_s = sub[sub['seed'] == seed]
                for omega in [1, 2, 4, 8, 16]:
                    try:
                        r_R = rows_s[rows_s['regime'] == 'reward-only'].iloc[0]
                        r_D = rows_s[rows_s['regime'] == 'duel-only'].iloc[0]
                        r_F = rows_s[rows_s['regime'] == 'fusion'].iloc[0]
                    except Exception:
                        continue
                    C_R = omega * r_R['T_r'] + 1 * r_R['T_d']
                    C_D = omega * r_D['T_r'] + 1 * r_D['T_d']
                    C_f = omega * r_F['T_r'] + 1 * r_F['T_d']
                    ratio = C_f / min(C_R, C_D) if min(C_R, C_D) > 0 else np.nan
                    fr.append(dict(case_name=case, delta=delta, seed=seed, omega_R=omega, C_R=C_R, C_D=C_D, C_f=C_f, ratio=ratio, fusion_wins=float(ratio < 1), success_R=r_R.get('success', False), success_D=r_D.get('success', False), success_f=r_F.get('success', False), T_R=r_R['T_r'], T_D=r_D['T_d'], T_r_f=r_F['T_r'], T_d_f=r_F['T_d'], T_f=r_F['T_total'], p_D_f=r_F.get('p_D', 0.0)))
    frdf = pd.DataFrame(fr)
    frdf.to_csv(os.path.join(outdir, 'fusion_cost_ratio.csv'), index=False)

    # summary by case/regime/delta
    # use 'error' as failure indicator; some older CSVs may not have 'error' so fallback to inverse of 'success'
    if 'error' in df.columns:
        fail_indicator = df['error']
    else:
        fail_indicator = ~df['success']
    df = df.assign(_error=fail_indicator)
    summ = df.groupby(['case_name', 'regime', 'delta']).agg(success_rate=('success', 'mean'), error_rate=('_error', 'mean'), median_T_r=('T_r', 'median'), q25_T_r=('T_r', lambda x: x.quantile(0.25)), q75_T_r=('T_r', lambda x: x.quantile(0.75)), median_T_d=('T_d', 'median'), q25_T_d=('T_d', lambda x: x.quantile(0.25)), q75_T_d=('T_d', lambda x: x.quantile(0.75)), median_T_total=('T_total', 'median'), q25_T_total=('T_total', lambda x: x.quantile(0.25)), q75_T_total=('T_total', lambda x: x.quantile(0.75)), median_p_D=('p_D', 'median'), q25_p_D=('p_D', lambda x: x.quantile(0.25)), q75_p_D=('p_D', lambda x: x.quantile(0.75))).reset_index()
    summ.to_csv(os.path.join(outdir, 'summary_by_case_regime_delta.csv'), index=False)

    # summary ratio by case/delta/omega
    s2 = frdf.groupby(['case_name', 'delta', 'omega_R']).agg(median_ratio=('ratio', 'median'), q25_ratio=('ratio', lambda x: x.quantile(0.25)), q75_ratio=('ratio', lambda x: x.quantile(0.75)), fusion_win_rate=('fusion_wins', 'mean'), median_C_R=('C_R', 'median'), median_C_D=('C_D', 'median'), median_C_f=('C_f', 'median')).reset_index()
    s2.to_csv(os.path.join(outdir, 'summary_ratio_by_case_delta_omega.csv'), index=False)
    print('Analysis outputs written to', outdir)
