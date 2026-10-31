"""Generate the 141-run discovery campaign and data-driven follow-up campaigns."""
import argparse,hashlib,json
from statistics import fmean
from pathlib import Path

def base(task,backbone,mode,seed):
    return dict(task=task,backbone=backbone,mode=mode,seed=seed,T=8,width=32,alpha=.8,theta=1.,
                gain='unit',detach_reset=False,epochs=80,lr=.001,motion_parents=6000,protocol='discovery_20261030')

def discovery():
    jobs=[]
    modes=['relu','leaky_relu','lif_hard','lif_soft','if_hard','if_soft','alif','plif','hetero']
    for task,backbones in [('cifar_classification',['conv','rescnn']),('cifar_reconstruction',['unet']),('motion_classification',['conv','rescnn'])]:
        for backbone in backbones:
            for mode in modes:
                for seed in [0,1,2]:jobs.append(base(task,backbone,mode,seed))
    for backbone in ['gru','tcn']:
        for seed in [0,1,2]:jobs.append(base('motion_classification',backbone,'leaky_relu',seed))
    return dict(stage='discovery',jobs=jobs,expected_runs=len(jobs),
                purpose='Find systematic phenomena across dynamics, architecture and task; retain every result and checkpoint.')

def smoke():
    jobs=[]
    for task,bb,mode in [('digits_classification','conv','lif_hard'),('digits_reconstruction','unet','lif_soft'),
                         ('motion_classification','gru','leaky_relu')]:
        j=base(task,bb,mode,0);j.update(width=4,epochs=2,motion_parents=80,protocol='smoke_only');jobs.append(j)
    return dict(stage='smoke_only',jobs=jobs,expected_runs=len(jobs))

def followup(results):
    records=[json.loads(p.read_text()) for p in Path(results).glob('*/complete.json')]
    eligible=[r for r in records if r['mode'] in ['lif_hard','lif_soft','alif','plif','hetero','leaky_relu'] and r['backbone'] in ['mlp','conv','rescnn','unet']]
    fields=['task','backbone','mode','T','width','alpha','theta','gain','detach_reset','epochs','lr','motion_parents','effective_batch','precision','implementation_sha']
    groups={}
    for r in eligible:groups.setdefault(tuple(r.get(k) for k in fields),[]).append(r)
    averaged=[]
    for group in groups.values():
        exemplar=dict(group[0]);exemplar['selection_source_cases']=[r['case_id'] for r in group]
        exemplar['selection_mean_reset_effect']=fmean(r.get('reset_effects',{}).get('no_reset_site_rate_calibrated',0) for r in group)
        averaged.append(exemplar)
    selected=[]
    for task in sorted(set(r['task'] for r in averaged)):
        candidates=[r for r in averaged if r['task']==task]
        candidates.sort(key=lambda r:abs(r['selection_mean_reset_effect']),reverse=True)
        if candidates:selected.append(candidates[0])
    if not selected:raise RuntimeError('No completed discovery cases available for follow-up selection.')
    jobs=[]
    for r in selected:
        for T,a,theta in [(t,a,1.) for t in [4,8,16] for a in [.5,.8,.95]]+[(8,.8,.5),(8,.8,2.)]:
            for seed in [3,4,5,6,7]:
                j=base(r['task'],r['backbone'],r['mode'],seed)
                j.update(T=T,alpha=a,theta=theta,protocol='new_seed_followup_20261030',selected_from=r['selection_source_cases'],selection_mean_reset_effect=r['selection_mean_reset_effect']);jobs.append(j)
    return dict(stage='new_seed_followup',selected_from=[r['selection_source_cases'] for r in selected],jobs=jobs,expected_runs=len(jobs),
                selection='Average calibrated no-reset effects across completed initialization seeds within each setting; select the largest absolute setting mean within each task; execute new initialization seeds and fixed sensitivity grid.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--follow-up-from');ap.add_argument('--output');args=ap.parse_args()
    if args.follow_up_from:
        obj=followup(args.follow_up_from);target=args.output or str(Path(__file__).resolve().parent/'campaigns/followup.json')
    else:
        obj=discovery();target=args.output or str(Path(__file__).resolve().parent/'campaigns/discovery.json')
        Path(target).parent.mkdir(parents=True,exist_ok=True)
        Path(target).with_name('smoke.json').write_text(json.dumps(smoke(),indent=2))
    Path(target).parent.mkdir(parents=True,exist_ok=True)
    Path(target).write_text(json.dumps(obj,indent=2));print(target,obj['expected_runs'])

