"""Windows/Linux CUDA queue. FP32 dynamics, adaptive microbatches, epoch resume."""
import argparse,copy,csv,hashlib,json,math,os,random,time,traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from models import ResearchNet,per_example_loss
from data import StudyData

PROJECT_ROOT=Path(__file__).resolve().parent.parent

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def case_id(job):
    return job['task']+'_'+job['backbone']+'_'+job['mode']+'_s'+str(job['seed'])+'_'+hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:10]

def source_fingerprint():
    root=Path(__file__).resolve().parent
    h=hashlib.sha256()
    for name in ['models.py','data.py','run_campaign.py']:
        h.update(name.encode());h.update((root/name).read_bytes())
    return h.hexdigest()

def device_info(device,min_gib=0):
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA is not exposed to this Python environment.')
        p=torch.cuda.get_device_properties(device)
        if p.total_memory/1024**3<min_gib:raise RuntimeError(f'GPU reports {p.total_memory/1024**3:.2f} GiB, below configured {min_gib} GiB.')
        free,total=torch.cuda.mem_get_info(device)
        return dict(device=str(device),name=p.name,total_gib=p.total_memory/1024**3,free_gib=free/1024**3,torch=torch.__version__,cuda_runtime=torch.version.cuda)
    return dict(device='cpu',torch=torch.__version__,purpose='explicit smoke/verification mode')

def make_loader(ds,batch,indices=None,shuffle=False,seed=0,workers=0):
    source=ds if indices is None else Subset(ds,indices)
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(source,batch_size=batch,shuffle=shuffle,generator=generator,num_workers=workers,pin_memory=torch.cuda.is_available(),persistent_workers=False)

def balanced_indices(ds,n,seed=713):
    rng=np.random.default_rng(seed);classes=np.unique(ds.labels);out=[]
    for k in classes:
        ids=np.flatnonzero(ds.labels==k);rng.shuffle(ids);out.extend(ids[:math.ceil(n/len(classes))].tolist())
    rng.shuffle(out);return np.array(out[:min(n,len(ds))],dtype=int)

def amp_context(device,precision):
    return torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=precision=='bf16' and device.type=='cuda')

def train_batch(net,batch,opt,device,micro,precision):
    n=len(batch['x']);recon=net.reconstruction
    # All microbatches contribute to one logical optimizer step, preserving effective batch size.
    while True:
        opt.zero_grad(set_to_none=True);value=0.
        try:
            for start in range(0,n,micro):
                x=batch['x'][start:start+micro].to(device,non_blocking=True);y=batch['y'][start:start+micro].to(device,non_blocking=True)
                with amp_context(device,precision):
                    pred,_=net(x);loss=per_example_loss(pred,y,recon).sum()/n
                loss.backward();value+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            return value,micro
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            if device.type!='cuda' or micro==1:raise
            pred=loss=x=y=None
            micro=max(1,micro//2);torch.cuda.empty_cache()
            print('OOM: retrying same optimizer batch with microbatch',micro,flush=True)

@torch.no_grad()
def evaluate(net,loader,device,iv=None,save_predictions=False):
    losses=[];scores=[];predictions=[];ids=[];iv=iv or {}
    for b in loader:
        x=b['x'].to(device,non_blocking=True);y=b['y'].to(device,non_blocking=True)
        if iv.get('reverse'):x=x.flip(1)
        if iv.get('swap_label'):y=1-y
        out,_=net(x,intervention=iv);loss=per_example_loss(out,y,net.reconstruction)
        score=-10*torch.log10(loss.clamp_min(1e-12)) if net.reconstruction else (out.argmax(1)==y).float()
        losses.extend(loss.cpu().tolist());scores.extend(score.cpu().tolist());ids.extend(b['id'].tolist())
        if save_predictions:predictions.append((out.sigmoid() if net.reconstruction else out).cpu().numpy())
    result=dict(loss=float(np.mean(losses)),score=float(np.mean(scores)),per_loss=np.array(losses),per_score=np.array(scores),ids=np.array(ids))
    if predictions:result['predictions']=np.concatenate(predictions)
    return result

@torch.no_grad()
def rates(net,loader,device,iv=None):
    total=0;acc=None
    for b in loader:
        x=b['x'].to(device);_,records=net(x,capture='rates',intervention=iv)
        if acc is None:acc=[torch.zeros_like(r['rate']) for r in records]
        for j,r in enumerate(records):acc[j]+=r['rate']*len(x)
        total+=len(x)
    return [v/total for v in (acc or [])]

def pooled(z):
    if z.ndim==5:
        n,t,c,h,w=z.shape;z=torch.nn.functional.adaptive_avg_pool2d(z.reshape(n*t,c,h,w),(min(4,h),min(4,w))).reshape(n,t,-1)
    return z

@torch.no_grad()
def retain_features(net,ds,indices,device,outdir,split,raw_n=0,batch=8):
    loader=make_loader(ds,batch,indices);outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    pool={};extra={k:[] for k in ['id','parent','label','nuisance','target']};saved=0
    for bi,b in enumerate(loader):
        x=b['x'].to(device);out,records=net(x,capture=True)
        for k in ['id','parent','label','nuisance']:extra[k].append(b[k].numpy())
        extra['target'].append(b['y'].numpy())
        for l,record in enumerate(records):
            for key in ['I','u','v','s','theta']:
                z=record[key];pool.setdefault(f'l{l}_{key}',[]).append(pooled(z).cpu().numpy())
            if saved<raw_n:
                keep=min(raw_n-saved,len(x));payload={k:v[:keep].cpu().numpy() for k,v in record.items()}
                if net.mode not in ['relu','leaky_relu'] and net.backbone not in ['gru','tcn']:payload['s']=payload['s'].astype('uint8')
                payload['ids']=b['id'][:keep].numpy()
                np.savez_compressed(outdir/f'{split}_raw_layer{l}_chunk{bi:04}.npz',**payload)
        if saved<raw_n:
            keep=min(raw_n-saved,len(x));np.savez_compressed(outdir/f'{split}_inputs_chunk{bi:04}.npz',
                x=b['x'][:keep].numpy(),target=b['y'][:keep].numpy(),mask=b['mask'][:keep].numpy(),ids=b['id'][:keep].numpy())
            saved+=keep
    result={k:np.concatenate(v) for k,v in pool.items()};result.update({k:np.concatenate(v) for k,v in extra.items()})
    np.savez_compressed(outdir/f'{split}_pooled_states.npz',**result)
    return result

def ridge_score(a,yt,b,yv,c,yc,classification):
    # Occupancy filtering and float64 algebra address rare-coordinate instability found in the pilot.
    mu=a.mean(0);std=a.std(0);keep=(std>max(float(np.median(std[std>1e-6]))*.01,1e-4)) if np.any(std>1e-6) else np.ones(a.shape[1],bool)
    keep=keep & ((np.abs(a-mu)>1e-5).mean(0)>=.01)
    scale=np.maximum(std[keep],1e-3)
    a=(a[:,keep].astype('float64')-mu[keep])/scale;b=(b[:,keep]-mu[keep])/scale;c=(c[:,keep]-mu[keep])/scale
    if classification:
        classes=np.unique(yt);yy=(yt[:,None]==classes).astype(float)
    else:yy=yt.reshape(len(yt),-1)
    ym=yy.mean(0);eig,u=np.linalg.eigh(a@a.T);proj=u.T@(yy-ym)
    kb=b@a.T;kc=c@a.T;best=None
    for alpha in [.1,1.,10.,100.,1000.]:
        coef=u@(proj/(np.maximum(eig,0)+alpha)[:,None]);pred=kb@coef+ym
        score=float(np.mean(classes[pred.argmax(1)]==yv)) if classification else -float(np.mean((np.clip(pred,0,1)-yv.reshape(len(yv),-1))**2))
        if best is None or score>best[0]:best=(score,coef,alpha)
    pred=kc@best[1]+ym
    score=float(np.mean(classes[pred.argmax(1)]==yc)) if classification else float(1-np.sum((np.clip(pred,0,1)-yc.reshape(len(yc),-1))**2)/max(np.sum((yc.reshape(len(yc),-1)-yc.reshape(len(yc),-1).mean(0))**2),1e-12))
    return score,best[2]

@torch.no_grad()
def paired_response(net,loader,device,iv=None,basis=None,eps=0.):
    sums=None;total=0;loss0=loss1=score0=score1=0.
    for b in loader:
        x=b['x'].to(device);y=b['y'].to(device);out0,r0=net(x,capture=True)
        xx=x
        if basis is not None:xx=x+eps*basis.to(x)[None]
        out1,r1=net(xx,capture=True,intervention=iv);n=len(x)
        l0=per_example_loss(out0,y,net.reconstruction);l1=per_example_loss(out1,y,net.reconstruction)
        loss0+=float(l0.sum());loss1+=float(l1.sum())
        score0+=float((-10*torch.log10(l0.clamp_min(1e-12))).sum()) if net.reconstruction else float((out0.argmax(1)==y).sum())
        score1+=float((-10*torch.log10(l1.clamp_min(1e-12))).sum()) if net.reconstruction else float((out1.argmax(1)==y).sum())
        if sums is None:sums=[dict(u=0.,v=0.,s=0.,rate=0.) for _ in r0]
        for l,(a,z) in enumerate(zip(r0,r1)):
            for key in ['u','v','s']:sums[l][key]+=float((a[key]-z[key]).abs().mean())*n
            sums[l]['rate']+=float(z['s'].mean())*n
        total+=n
    return [dict(layer=l,delta_loss=(loss1-loss0)/total,delta_score=(score1-score0)/total,
                 delta_u=s['u']/total,delta_v=s['v']/total,delta_s=s['s']/total,mean_emission=s['rate']/total) for l,s in enumerate(sums or [])]

def audit(net,sets,device,folder,args):
    import pandas as pd
    folder=Path(folder);features={}
    for split,n in [('train',args.probe_train),('val',args.probe_val),('test',args.probe_test)]:
        ds=sets[split];ids=balanced_indices(ds,n);features[split]=retain_features(net,ds,ids,device,folder/'states',split,
            raw_n=args.raw_state_samples if split=='test' else 0,batch=min(8,args.eval_batch))
    probes=[]
    for key in features['train']:
        if not key.endswith(('_u','_v','_s')):continue
        for readout in ['mean','last']:
            xx={sp:(d[key].mean(1) if readout=='mean' else d[key][:,-1]) for sp,d in features.items()}
            for target,cls in [('target',not net.reconstruction),('nuisance',True)]:
                yy={sp:d[target] for sp,d in features.items()}
                score,alpha=ridge_score(xx['train'],yy['train'],xx['val'],yy['val'],xx['test'],yy['test'],cls)
                probes.append(dict(feature=key,readout=readout,target=target,score=score,regularization=alpha))
    pd.DataFrame(probes).to_csv(folder/'probes.csv',index=False)
    cal=make_loader(sets['train'],min(16,args.eval_batch),balanced_indices(sets['train'],args.calibration_samples))
    test=make_loader(sets['test'],args.eval_batch)
    base=evaluate(net,test,device,save_predictions=True)
    np.savez_compressed(folder/'test_predictions.npz',**base)
    interventions=[dict(condition='baseline',score=base['score'],loss=base['loss'],delta_score=0.,delta_loss=0.)]
    def add(name,iv):
        res=evaluate(net,test,device,iv);interventions.append(dict(condition=name,score=res['score'],loss=res['loss'],delta_score=res['score']-base['score'],delta_loss=res['loss']-base['loss']))
    add('alpha_zero',{'alpha':0.})
    add('reverse_time',{'reverse':True})
    if sets['test'].motion:add('reverse_time_correct_label',{'reverse':True,'swap_label':True})
    for l in range(len(net.cells)):add(f'erase_layer_{l}',{'erase_layer':l})
    rate_rows=[]
    if net.cells and net.mode not in ['relu','leaky_relu']:
        add('no_reset_raw',{'reset':'none'})
        target=rates(net,cal,device);overrides={}
        for l,t in enumerate(target):
            lo=torch.full_like(t,.0001);hi=torch.full_like(t,200.)
            for _ in range(12):
                overrides[l]=(lo+hi)/2
                rr=rates(net,cal,device,{'reset':'none','theta_overrides':overrides})[l]
                too_many=rr>t;lo=torch.where(too_many,overrides[l],lo);hi=torch.where(too_many,hi,overrides[l])
            overrides[l]=(lo+hi)/2
        achieved=rates(net,cal,device,{'reset':'none','theta_overrides':overrides})
        for l,(t,a) in enumerate(zip(target,achieved)):rate_rows.append(dict(layer=l,calibration_mae=float(abs(a-t).mean()),calibration_max_error=float(abs(a-t).max())))
        torch.save({k:v.cpu() for k,v in overrides.items()},folder/'site_threshold_overrides.pt')
        add('no_reset_site_rate_calibrated',{'reset':'none','theta_overrides':overrides})
    pd.DataFrame(interventions).to_csv(folder/'interventions.csv',index=False)
    if rate_rows:pd.DataFrame(rate_rows).to_csv(folder/'rate_calibration.csv',index=False)
    if args.frequency_map:
        ds=sets['test'];small=make_loader(ds,min(8,args.eval_batch),balanced_indices(ds,args.perturbation_samples));rows=[]
        T=ds.T;_,h,w=ds.shape;tt=torch.arange(T)[:,None,None];yy=torch.arange(h)[None,:,None];xx=torch.arange(w)[None,None,:]
        for k in sorted(set([0,1,max(1,w//3)])):
            for omega in sorted(set([0,1,max(1,T//2-1)])):
                p=torch.cos(2*torch.pi*(k*xx/w+omega*tt/T)).expand(T,h,w).clone();p/=p.square().mean().sqrt();p=p[:,None].expand(T,ds.shape[0],h,w)
                for eps in [.05,.15]:
                    for sign in [-1,1]:
                        for row in paired_response(net,small,device,basis=p,eps=eps*sign):
                            row.update(spatial_bin=k,temporal_bin=omega,epsilon_rms=eps,sign=sign);rows.append(row)
        pd.DataFrame(rows).to_csv(folder/'frequency_map.csv',index=False)
    return base,interventions

def run_job(job,args,device):
    folder=Path(args.out)/case_id(job);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return json.loads((folder/'complete.json').read_text())
    seed=job['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats(device)
    sets={sp:StudyData(job['task'],sp,T=job['T'],root=args.data_root,motion_parents=job.get('motion_parents',6000)) for sp in ['train','val','test']}
    if args.smoke_samples:
        for ds in sets.values():
            n=min(args.smoke_samples,len(ds));ds.ids=ds.ids[:n];ds.parents=ds.parents[:n];ds.labels=ds.labels[:n]
            if ds.clean is not None:ds.clean=ds.clean[:n]
    for sp,ds in sets.items():atomic_json(folder/f'data_{sp}.json',ds.manifest())
    net=ResearchNet(sets['train'].shape,job['task'],job['backbone'],job['mode'],job['width'],job['alpha'],job['theta'],job.get('gain','unit'),job.get('detach_reset',False)).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=job.get('lr',.001),weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,job['epochs'])
    start_epoch=0;best=math.inf;micro=args.microbatch;last=folder/'last.pt';history=[]
    if last.exists():
        c=torch.load(last,map_location=device,weights_only=False);net.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler'])
        start_epoch=c['epoch'];best=c['best'];micro=c['microbatch'];history=c['history']
        torch.set_rng_state(c['torch_rng'].cpu())
        if device.type=='cuda' and c.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([s.cpu() for s in c['cuda_rng']])
    atomic_json(folder/'config.json',dict(job=job,case_id=case_id(job),device=device_info(device),effective_batch=args.batch,parameter_count=sum(p.numel() for p in net.parameters()),precision=args.precision))
    t0=time.time()
    for epoch in range(start_epoch,job['epochs']):
        net.train();sets['train'].epoch=epoch;total=0.;count=0
        train=make_loader(sets['train'],args.batch,shuffle=True,seed=seed+10000*epoch,workers=args.workers)
        for b in train:
            value,micro=train_batch(net,b,opt,device,micro,args.precision);total+=value*len(b['x']);count+=len(b['x'])
        net.eval();val=evaluate(net,make_loader(sets['val'],args.eval_batch),device)
        sched.step();row=dict(epoch=epoch+1,train_loss=total/count,val_loss=val['loss'],val_score=val['score'],microbatch=micro)
        history.append(row)
        if val['loss']<best:
            best=val['loss'];torch.save({'model':net.state_dict(),'job':job,'epoch':epoch+1},folder/'best.pt')
        snapshot=dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch+1,best=best,microbatch=micro,history=history,
                      torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        tmp=folder/'last.tmp';torch.save(snapshot,tmp);tmp.replace(last)
        if epoch+1 in [1,5,10,20,40,80,100]:torch.save({'model':net.state_dict(),'job':job,'epoch':epoch+1},folder/f'epoch_{epoch+1:03}.pt')
        atomic_json(folder/'history.json',history)
        print(case_id(job),row,flush=True)
    net.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=False)['model']);net.eval()
    for ds in sets.values():ds.epoch=0
    if args.no_audit:
        result=evaluate(net,make_loader(sets['test'],args.eval_batch),device,save_predictions=True)
        np.savez_compressed(folder/'test_predictions.npz',**result);iv=[]
    else:result,iv=audit(net,sets,device,folder,args)
    summary=dict(**job,case_id=case_id(job),test_score=result['score'],test_loss=result['loss'],seconds_this_invocation=time.time()-t0,
                 peak_vram_gib=torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0,device=device_info(device))
    if iv:summary['reset_effects']={r['condition']:r['delta_score'] for r in iv}
    atomic_json(folder/'complete.json',summary);return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',default=str(PROJECT_ROOT/'src/campaigns/discovery.json'));ap.add_argument('--out',default=str(PROJECT_ROOT/'results/discovery'))
    ap.add_argument('--data-root',default=str(PROJECT_ROOT/'data'));ap.add_argument('--device',default='cuda:0');ap.add_argument('--min-vram-gib',type=float,default=0)
    ap.add_argument('--batch',type=int,default=256);ap.add_argument('--microbatch',type=int,default=64);ap.add_argument('--eval-batch',type=int,default=16)
    ap.add_argument('--workers',type=int,default=0);ap.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    ap.add_argument('--probe-train',type=int,default=512);ap.add_argument('--probe-val',type=int,default=256);ap.add_argument('--probe-test',type=int,default=1024)
    ap.add_argument('--raw-state-samples',type=int,default=64);ap.add_argument('--calibration-samples',type=int,default=128)
    ap.add_argument('--perturbation-samples',type=int,default=96);ap.add_argument('--frequency-map',action='store_true')
    ap.add_argument('--limit',type=int,default=0);ap.add_argument('--max-hours',type=float,default=0);ap.add_argument('--no-audit',action='store_true')
    ap.add_argument('--smoke-samples',type=int,default=0);ap.add_argument('--epochs-override',type=int,default=0);ap.add_argument('--allow-cpu',action='store_true')
    ap.add_argument('--preflight',action='store_true')
    args=ap.parse_args();torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(args.device)
    if device.type!='cuda' and not args.allow_cpu:raise RuntimeError('Use CUDA for the research campaign; --allow-cpu explicitly enables smoke verification.')
    info=device_info(device,args.min_vram_gib);print(json.dumps(info,indent=2),flush=True)
    if args.preflight:return
    if args.precision=='bf16' and device.type=='cuda' and not torch.cuda.is_bf16_supported():raise RuntimeError('This device/runtime does not support bf16.')
    manifest=json.loads(Path(args.campaign).read_text());jobs=manifest['jobs']
    if args.limit:jobs=jobs[:args.limit]
    if args.epochs_override:
        jobs=[dict(j,epochs=args.epochs_override,protocol='explicit_smoke_override') for j in jobs]
    if args.smoke_samples:jobs=[dict(j,smoke_samples=args.smoke_samples) for j in jobs]
    implementation=source_fingerprint()
    jobs=[dict(j,implementation_sha=implementation,effective_batch=args.batch,precision=args.precision,execution_backend=device.type,torch_version=torch.__version__) for j in jobs]
    Path(args.out).mkdir(parents=True,exist_ok=True);atomic_json(Path(args.out)/'invocation.json',vars(args))
    atomic_json(Path(args.out)/'requested_cases.json',dict(campaign=str(Path(args.campaign).resolve()),implementation_sha=implementation,jobs=jobs,case_ids=[case_id(j) for j in jobs]))
    started=time.time()
    for job in jobs:
        if args.max_hours and (time.time()-started)/3600>=args.max_hours:break
        try:run_job(job,args,device)
        except KeyboardInterrupt:raise
        except Exception as exc:
            folder=Path(args.out)/case_id(job);atomic_json(folder/'failed.json',dict(job=job,error=str(exc),traceback=traceback.format_exc()))
            print('FAILED',case_id(job),str(exc),flush=True)
            raise
    summaries=[json.loads(p.read_text()) for p in Path(args.out).glob('*/complete.json')]
    atomic_json(Path(args.out)/'completed_summary.json',summaries)
    print('COMPLETED_CASES',len(summaries),flush=True)

if __name__=='__main__':main()

