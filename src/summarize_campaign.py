"""Aggregate completed GPU cases into task-conditioned phenomena and readable reports."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',required=True);args=ap.parse_args();root=Path(args.runs)
    rows=[];probes=[];ivs=[];freq=[]
    for p in sorted(root.glob('*/complete.json')):
        s=json.loads(p.read_text());row={k:v for k,v in s.items() if not isinstance(v,(dict,list))}
        rows.append(row)
        for name,collection in [('probes.csv',probes),('interventions.csv',ivs),('frequency_map.csv',freq)]:
            f=p.parent/name
            if f.exists():
                d=pd.read_csv(f)
                for k in ['case_id','task','backbone','mode','seed']:d[k]=s[k]
                collection.append(d)
    if not rows:raise RuntimeError('No completed cases')
    run=pd.DataFrame(rows);run.to_csv(root/'all_runs.csv',index=False)
    for name,collection in [('all_probes.csv',probes),('all_interventions.csv',ivs),('all_frequency_maps.csv',freq)]:
        if collection:pd.concat(collection,ignore_index=True).to_csv(root/name,index=False)
    stats=run.groupby(['task','backbone','mode'])['test_score'].agg(['count','mean','std']).reset_index()
    stats.to_csv(root/'performance_by_setting.csv',index=False)
    sections=['# Stateflow discovery: completed results',f'Completed trained cases: {len(run)}','',
              '## Performance by setting','',stats.to_string(index=False),'']
    if ivs:
        d=pd.concat(ivs,ignore_index=True)
        t=d.groupby(['task','backbone','mode','condition'])[['delta_score','delta_loss']].agg(['mean','std'])
        t.to_csv(root/'interventions_by_setting.csv');sections.extend(['## Frozen interventions','',t.to_string(),''])
    if probes:
        d=pd.concat(probes,ignore_index=True)
        p=d.groupby(['task','backbone','mode','feature','target','readout']).score.agg(['mean','std'])
        p.to_csv(root/'probes_by_setting.csv')
    if freq:
        d=pd.concat(freq,ignore_index=True)
        nonzero=d[d.delta_s>.001].copy();nonzero['functional_change_per_activity_change']=abs(nonzero.delta_loss)/nonzero.delta_s
        priority=nonzero.sort_values('functional_change_per_activity_change',ascending=False).groupby('task').head(20)
        priority.to_csv(root/'functional_activity_dissociations.csv',index=False)
        sections.extend(['## Strong functional/activity dissociations','',
                         priority[['task','backbone','mode','layer','spatial_bin','temporal_bin','delta_s','delta_loss','functional_change_per_activity_change']].to_string(index=False),''])
    (root/'campaign_analysis.md').write_text('\n'.join(sections),encoding='utf-8')
    print('Aggregated',len(run),'completed cases into',root)

if __name__=='__main__':main()

