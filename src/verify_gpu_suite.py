"""Scientific implementation checks, no external dataset download."""
import copy,json,tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import StudyData
from models import ResearchNet,MODES
from run_campaign import train_batch,case_id
torch.set_num_threads(2)

def main():
    checks={};torch.manual_seed(13)
    for backbone in ['mlp','conv','rescnn','unet','gru','tcn']:
        modes=MODES if backbone in ['mlp','conv','unet'] else ['lif_hard','lif_soft','leaky_relu']
        for mode in modes:
            task='digits_reconstruction' if backbone=='unet' else 'digits_classification'
            model=ResearchNet((1,8,8),task,backbone,mode,width=4)
            x=torch.randn(2,4,1,8,8)
            out,rec=model(x,capture=True)
            assert tuple(out.shape)==((2,1,8,8) if backbone=='unet' else (2,10))
            loss=out.square().mean();loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if backbone in ['mlp','conv'] and mode=='lif_soft':
                r=rec[0];prev=torch.cat([torch.zeros_like(r['v'][:,:1]),r['v'][:,:-1]],1)
                err=float((r['I']-(r['theta']*r['s']+r['v']-.8*prev)).abs().max())
                assert err<1e-5;checks['soft_identity_'+backbone]=err
            checks['finite_grad_'+backbone+'_'+mode]=True
    for backbone in ['mlp','conv','rescnn','unet']:
        task='digits_reconstruction' if backbone=='unet' else 'digits_classification'
        m=ResearchNet((1,8,8),task,backbone,'relu',width=4);x=torch.randn(3,8,1,8,8)
        with torch.no_grad():err=float((m(x)[0]-m(x.flip(1))[0]).abs().max())
        assert err<1e-5;checks['relu_reversal_'+backbone]=err
    for T in [4,8,16]:
        sets={sp:StudyData('motion_classification',sp,T,motion_parents=100) for sp in ['train','val','test']}
        assert not set(sets['train'].parents)&set(sets['test'].parents)
        d=sets['test']
        parent=int(d.parents[0]);j=int(np.flatnonzero(d.parents==parent)[0])
        a,b=d[j],d[j+1]
        assert torch.equal(a['x'],b['x'].flip(0))
        checks['exact_frame_pair_T'+str(T)]=True
    # Logical-batch update equals microbatch accumulation at fixed parameters.
    torch.manual_seed(41);a=ResearchNet((1,8,8),'digits_classification','mlp','lif_soft',width=4);b=copy.deepcopy(a)
    batch={'x':torch.randn(4,4,1,8,8),'y':torch.tensor([0,1,2,3])}
    oa=torch.optim.SGD(a.parameters(),lr=.01);ob=torch.optim.SGD(b.parameters(),lr=.01)
    train_batch(a,batch,oa,torch.device('cpu'),4,'fp32');train_batch(b,batch,ob,torch.device('cpu'),1,'fp32')
    err=max(float((x-y).abs().max()) for x,y in zip(a.parameters(),b.parameters()))
    assert err<2e-6;checks['microbatch_update_error']=err
    assert case_id({'task':'x','backbone':'conv','mode':'lif_hard','seed':0,'T':4})!=case_id({'task':'x','backbone':'conv','mode':'lif_hard','seed':0,'T':8})
    checks['config_hash_distinguishes_settings']=True
    for backbone,task in [('conv','cifar_classification'),('rescnn','cifar_classification'),('unet','cifar_reconstruction')]:
        m=ResearchNet((3,32,32),task,backbone,'lif_soft',width=4)
        out,_=m(torch.randn(2,4,3,32,32));out.square().mean().backward()
        assert tuple(out.shape)==((2,3,32,32) if backbone=='unet' else (2,10))
        checks['RGB32_forward_backward_'+backbone]=True
    Path('verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()

