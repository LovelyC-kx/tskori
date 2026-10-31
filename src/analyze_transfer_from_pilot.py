"""Additional analysis on existing executed results: predict functional effects in an unseen topology."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score,mean_absolute_error

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot-directory',required=True);args=ap.parse_args()
    root=Path(args.pilot_directory);df=pd.read_csv(root/'results/network_frequency_response.csv')
    key=['run','task','arch','mode','seed','spatial_bin','orientation','temporal_bin','epsilon_rms','sign']
    top=df[df.layer==2].copy()
    rows=[];predictions=[]
    for task in ['digits','motion','reconstruction']:
        q=top[(top.task==task)&top['mode'].isin(['leaky_relu','lif_hard'])].copy()
        base=np.c_[q.spatial_bin,q.temporal_bin,q.epsilon_rms,q.sign,(q.orientation=='y').astype(float),(q['mode']=='lif_hard').astype(float)]
        activity=np.c_[base,q.emitted_mean_rate,q.emitted_mean_abs_change]
        trajectory=[]
        for l in [0,1,2]:
            z=df[df.layer==l].set_index(key)
            cur=z.loc[pd.MultiIndex.from_frame(q[key])]
            trajectory.append(cur[['pre_reset_mean_abs_change','post_reset_mean_abs_change','emitted_mean_abs_change','emitted_mean_rate']].to_numpy())
        richer=np.c_[base,*trajectory]
        train=q.arch.to_numpy()=='plain';test=q.arch.to_numpy()=='residual';y=q.delta_loss.to_numpy()
        for name,x in [('input_structure',base),('plus_emitted_activity',activity),('plus_layerwise_state',richer)]:
            for seed in [17,29,47]:
                model=ExtraTreesRegressor(n_estimators=250,min_samples_leaf=3,max_features=1.,random_state=seed,n_jobs=2)
                model.fit(x[train],y[train]);pred=model.predict(x[test])
                rows.append(dict(task=task,features=name,predictor_seed=seed,train_topology='plain',test_topology='residual',
                                 test_r2=r2_score(y[test],pred),test_mae=mean_absolute_error(y[test],pred),
                                 n_train=int(train.sum()),n_test=int(test.sum())))
                predictions.extend(dict(task=task,features=name,predictor_seed=seed,run=q.iloc[idx]['run'],truth=y[idx],prediction=pr)
                                   for idx,pr in zip(np.flatnonzero(test),pred))
    pd.DataFrame(rows).to_csv('pilot_topology_transfer.csv',index=False)
    pd.DataFrame(predictions).to_csv('pilot_topology_transfer_predictions.csv',index=False)
    print(pd.DataFrame(rows).groupby(['task','features'])[['test_r2','test_mae']].mean().round(6).to_string())
    print('Predictor seeds repeat model fitting; independent SNN test runs are the retained residual-topology models.')

if __name__=='__main__':main()

