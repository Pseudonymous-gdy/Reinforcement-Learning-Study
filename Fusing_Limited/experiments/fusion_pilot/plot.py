import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def heatmap_ratio(summary_csv, outdir):
    df = pd.read_csv(summary_csv)
    os.makedirs(outdir, exist_ok=True)
    cases = df['case_name'].unique()
    for case in cases:
        sub = df[df['case_name'] == case]
        deltas = sorted(sub['delta'].unique())
        omegas = sorted(sub['omega_R'].unique())
        mat = np.zeros((len(deltas), len(omegas)))
        for i, delta in enumerate(deltas):
            for j, omega in enumerate(omegas):
                val = sub[(sub['delta'] == delta) & (sub['omega_R'] == omega)]['median_ratio']
                mat[i, j] = float(val.iloc[0]) if len(val) > 0 else np.nan
        plt.figure(figsize=(6, 3))
        plt.imshow(mat, cmap='RdYlBu_r', vmin=0.0, vmax=np.nanmax(mat))
        plt.colorbar()
        plt.xticks(np.arange(len(omegas)), omegas)
        plt.yticks(np.arange(len(deltas)), deltas)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center', color='k')
        plt.title(f'Heatmap ratio {case}')
        plt.xlabel('omega_R')
        plt.ylabel('delta')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'heatmap_ratio_{case}.png'))
        plt.close()
