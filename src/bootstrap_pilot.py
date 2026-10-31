"""Paired hierarchical bootstrap of retained per-neuron calibration predictions."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot-directory',required=True);args=ap.parse_args()
    root=Path(args.pilot_directory)/'results/neuron_rate_controls';rng=np.random.default_rng(530);rows=[]
    for task in ['digits','motion']:
        for mode in ['lif_hard','lif_soft']:
            deltas=[];base=[];controlled=[];parents0=None
            for seed in [0,1,2]:
                z=np.load(root/f'{task}_plain_{mode}_seed{seed}.npz')
                y=z['targets'];a=(z['original_logits'].argmax(1)==y).astype(float);b=(z['controlled_logits'].argmax(1)==y).astype(float)
                parents=z['parents'];unique=np.unique(parents)
                if parents0 is not None:assert np.array_equal(unique,parents0)
                parents0=unique
                deltas.append(np.array([(b[parents==p]-a[parents==p]).mean() for p in unique]))
                base.append(a.mean());controlled.append(b.mean())
            delta=np.array(deltas);boot=[]
            for _ in range(10000):
                ss=rng.integers(0,3,3);pp=rng.integers(0,delta.shape[1],delta.shape[1]);boot.append(delta[ss][:,pp].mean())
            lo,hi=np.quantile(boot,[.025,.975])
            rows.append(dict(task=task,mode=mode,seeds=3,test_parents=len(parents0),baseline_accuracy=np.mean(base),controlled_accuracy=np.mean(controlled),
                             delta_pp=delta.mean()*100,paired_hierarchical_95_low_pp=lo*100,paired_hierarchical_95_high_pp=hi*100,bootstrap_draws=10000))
    d=pd.DataFrame(rows);d.to_csv('pilot_reset_effect_intervals.csv',index=False);print(d.to_string(index=False))

if __name__=='__main__':main()

