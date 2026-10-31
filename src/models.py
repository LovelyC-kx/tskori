"""CUDA-ready spatiotemporal models. State updates use float32 throughout."""
import math
import torch
from torch import nn
from torch.nn import functional as F

MODES=['relu','leaky_relu','lif_hard','lif_soft','if_hard','if_soft','alif','plif','hetero']

class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return (x>=0).to(x.dtype)
    @staticmethod
    def backward(ctx,g):
        x,=ctx.saved_tensors
        return g/(1+2*x.abs()).square()

class Dynamics(nn.Module):
    def __init__(self,channels,mode,alpha=.8,theta=1.,gain='unit',detach_reset=False):
        super().__init__()
        self.mode=mode;self.channels=channels;self.gain=gain;self.detach_reset=detach_reset
        self.register_buffer('theta',torch.tensor(float(theta)))
        if mode=='plif':
            self.alpha_logit=nn.Parameter(torch.tensor(math.log(alpha/(1-alpha))))
        else:
            a=torch.linspace(max(.05,alpha-.15),min(.99,alpha+.15),channels) if mode=='hetero' else torch.tensor(alpha)
            self.register_buffer('alpha_value',a)
    def alpha(self,v,override=None):
        a=override if override is not None else (self.alpha_logit.sigmoid() if self.mode=='plif' else self.alpha_value)
        if self.mode.startswith('if_') and override is None:a=1.
        a=torch.as_tensor(a,device=v.device,dtype=v.dtype)
        return a.reshape(1,-1,*([1]*(v.ndim-2))) if a.ndim else a
    def forward(self,current,index,capture=False,intervention=None):
        iv=intervention or {};current=current.float()
        if iv.get('pulse_layer')==index:
            current=current.clone();current[:,iv.get('pulse_step',current.shape[1]//2)]+=iv['pulse_vector'].to(current)
        v=torch.zeros_like(current[:,0]);adapt=torch.zeros_like(v)
        alpha=self.alpha(v,iv.get('alpha'));outs=[];us=[];vs=[];ts=[]
        base_theta=iv.get('theta_overrides',{}).get(index,self.theta)
        for t in range(current.shape[1]):
            if iv.get('erase_layer')==index and t==current.shape[1]//2:
                v=torch.zeros_like(v);adapt=torch.zeros_like(adapt)
            theta=base_theta+(adapt if self.mode=='alif' else 0)
            if self.mode=='relu':u=current[:,t];s=F.relu(u);v=u
            else:
                gain=(1-alpha) if self.gain=='dc_normalized' and not self.mode.startswith('if_') else 1.
                u=alpha*v+gain*current[:,t]
                if self.mode=='leaky_relu':s=F.relu(u);v=u
                else:
                    s=Spike.apply(u-theta);sr=s.detach() if self.detach_reset else s
                    reset=iv.get('reset','hard' if self.mode in ['lif_hard','if_hard','hetero'] else 'soft')
                    if reset=='hard':v=u*(1-sr)
                    elif reset=='soft':v=u-theta*sr
                    elif reset=='none':v=u
                    else:raise ValueError(reset)
                    if self.mode=='alif':adapt=.9*adapt+.1*s
            outs.append(s)
            if capture is True:us.append(u);vs.append(v);ts.append(torch.as_tensor(theta,device=u.device).expand_as(u))
        out=torch.stack(outs,1)
        if capture=='rates':
            rec={'rate':out.mean((0,1)).detach()}
        elif capture is True:
            rec={'I':current,'u':torch.stack(us,1),'v':torch.stack(vs,1),'s':out,'theta':torch.stack(ts,1)}
        else:rec=None
        return out,rec

class Project(nn.Module):
    def __init__(self,inc,outc,mode,spatial=True,stride=1,**kw):
        super().__init__()
        self.op=nn.Conv2d(inc,outc,3,stride=stride,padding=1) if spatial else nn.Linear(inc,outc)
        self.norm=nn.GroupNorm(1,outc) if spatial else nn.LayerNorm(outc)
        self.cell=Dynamics(outc,mode,**kw);self.spatial=spatial
    def forward(self,x,index,capture=False,intervention=None,skip=False):
        if self.spatial:
            n,t,c,h,w=x.shape;y=self.norm(self.op(x.reshape(n*t,c,h,w)))
            y=y.reshape(n,t,*y.shape[1:])
        else:y=self.norm(self.op(x))
        if skip:y=y+.5*x
        return self.cell(y,index,capture,intervention)

class ResearchNet(nn.Module):
    def __init__(self,shape,task,backbone='rescnn',mode='lif_hard',width=32,alpha=.8,theta=1.,gain='unit',detach_reset=False):
        super().__init__()
        c,h,w=shape;self.shape=shape;self.task=task;self.backbone=backbone;self.mode=mode
        self.reconstruction=task.endswith('reconstruction');self.blocks=nn.ModuleList()
        kw=dict(alpha=alpha,theta=theta,gain=gain,detach_reset=detach_reset)
        nout=c*h*w if self.reconstruction else 2 if task.startswith('motion') else 10
        if backbone=='mlp':
            sizes=[c*h*w,width*4,width*4,width*4]
            self.blocks=nn.ModuleList([Project(sizes[j],sizes[j+1],mode,False,**kw) for j in range(3)])
            self.head=nn.Linear(sizes[-1],nout)
        elif backbone in ['conv','rescnn']:
            plan=[(c,width,1),(width,width,1),(width,width*2,2),(width*2,width*4,2)]
            self.blocks=nn.ModuleList([Project(a,b,mode,True,stride=s,**kw) for a,b,s in plan])
            self.head=nn.Linear(width*4,nout)
        elif backbone=='unet':
            plan=[(c,width,1),(width,width*2,2),(width*2,width*4,2),(width*6,width*2,1),(width*3,width,1)]
            self.blocks=nn.ModuleList([Project(a,b,mode,True,stride=s,**kw) for a,b,s in plan])
            self.head=nn.Conv2d(width,c,1)
        elif backbone=='gru':
            self.embedding=nn.Sequential(nn.Linear(c*h*w,width*4),nn.ReLU())
            self.gru=nn.GRU(width*4,width*4,batch_first=True,num_layers=2)
            self.head=nn.Linear(width*4,nout)
        elif backbone=='tcn':
            self.tcn=nn.ModuleList([nn.Conv3d(c,width,(3,3,3),padding=1),
                                     nn.Conv3d(width,width*2,(3,3,3),padding=1,stride=(1,2,2)),
                                     nn.Conv3d(width*2,width*4,(3,3,3),padding=1,stride=(1,2,2))])
            self.head=nn.Linear(width*4,nout)
        else:raise ValueError(backbone)
    @property
    def cells(self):return [b.cell for b in self.blocks]
    def forward(self,x,capture=False,intervention=None):
        records=[]
        if self.backbone=='gru':
            xx=self.embedding(x.flatten(2));y,_=self.gru(xx)
            if capture is True:records=[{'I':xx,'u':y,'v':y,'s':y,'theta':torch.zeros_like(y)}]
            out=self.head(y.mean(1))
        elif self.backbone=='tcn':
            y=x.transpose(1,2)
            for block in self.tcn:
                z=block(y);y=F.relu(z)
                if capture is True:
                    q=y.transpose(1,2);records.append({'I':z.transpose(1,2),'u':z.transpose(1,2),'v':q,'s':q,'theta':torch.zeros_like(q)})
            out=self.head(y.mean((2,3,4)))
        elif self.backbone=='unet':
            n,t=x.shape[:2];saved=[];y=x
            for l,b in enumerate(self.blocks):
                if l==3:
                    q=saved[1];y=F.interpolate(y.flatten(0,1),size=q.shape[-2:],mode='bilinear',align_corners=False).reshape(n,t,y.shape[2],*q.shape[-2:])
                    y=torch.cat([y,q],2)
                elif l==4:
                    q=saved[0];y=F.interpolate(y.flatten(0,1),size=q.shape[-2:],mode='bilinear',align_corners=False).reshape(n,t,y.shape[2],*q.shape[-2:])
                    y=torch.cat([y,q],2)
                y,r=b(y,l,capture,intervention)
                if l<3:saved.append(y)
                if r is not None:records.append(r)
            out=self.head(y.flatten(0,1)).reshape(n,t,*self.shape).mean(1)
        else:
            y=x.flatten(2) if self.backbone=='mlp' else x
            for l,b in enumerate(self.blocks):
                skip=self.backbone=='rescnn' and l==1
                y,r=b(y,l,capture,intervention,skip)
                if r is not None:records.append(r)
            pooled=y.mean(1)
            if pooled.ndim==4:pooled=pooled.mean((-2,-1))
            out=self.head(pooled)
        if self.reconstruction and out.ndim==2:out=out.reshape(len(x),*self.shape)
        return out,records

def per_example_loss(out,y,reconstruction):
    if reconstruction:return (out.sigmoid()-y).square().flatten(1).mean(1)
    return F.cross_entropy(out,y,reduction='none')

