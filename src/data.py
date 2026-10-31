"""Deterministic sample IDs, parent-group splits, original-image CIFAR splits."""
from pathlib import Path
import hashlib,pickle,tarfile,urllib.request,json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def cifar10(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);archive=root/'cifar-10-python.tar.gz'
    expected='c58f30108f718f92721af3b95e74349a'
    if not archive.exists():
        tmp=archive.with_suffix('.partial')
        urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',tmp);tmp.replace(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest()!=expected:raise RuntimeError('CIFAR archive checksum mismatch')
    cache=root/'cifar10_arrays.npz'
    if not cache.exists():
        arrays={};labels={}
        with tarfile.open(archive) as tar:
            for split,names in [('train',[f'data_batch_{i}' for i in range(1,6)]),('test',['test_batch'])]:
                xx=[];yy=[]
                for name in names:
                    blob=tar.extractfile('cifar-10-batches-py/'+name).read()
                    obj=pickle.loads(blob,encoding='bytes')
                    xx.append(obj[b'data'].reshape(-1,3,32,32));yy.extend(obj[b'labels'])
                arrays[split]=np.concatenate(xx);labels[split+'_labels']=np.array(yy,dtype='int64')
        temporary=cache.with_name('cifar10_arrays.partial.npz')
        np.savez_compressed(temporary,**arrays,**labels)
        temporary.replace(cache)
    with np.load(cache) as z:return {k:z[k] for k in z.files}

class StudyData(Dataset):
    def __init__(self,task,split,T=8,root='data',data_seed=101,motion_parents=6000,motion_size=16):
        self.task=task;self.split=split;self.T=T;self.seed=data_seed;self.epoch=0
        self.reconstruction=task.endswith('reconstruction');self.motion=task.startswith('motion')
        if self.motion:
            parents=np.arange(motion_parents)
            tr,o=train_test_split(parents,test_size=.3,random_state=2026)
            va,te=train_test_split(o,test_size=.5,random_state=29)
            ids={'train':tr,'val':va,'test':te}[split]
            self.ids=np.ravel(np.c_[ids*2,ids*2+1]);self.shape=(1,motion_size,motion_size)
            self.labels=self.ids%2;self.parents=self.ids//2
            self.clean=None
        elif task.startswith('digits'):
            d=load_digits();allx=(d.data/16).reshape(-1,1,8,8).astype('float32');labels=d.target
            ids=np.arange(len(labels));tr,o=train_test_split(ids,test_size=.33,stratify=labels,random_state=2026)
            va,te=train_test_split(o,test_size=.6,stratify=labels[o],random_state=29)
            self.ids={'train':tr,'val':va,'test':te}[split];self.clean=allx[self.ids];self.labels=labels[self.ids];self.parents=self.ids
            self.shape=(1,8,8)
        else:
            raw=cifar10(root)
            if split=='test':
                self.clean=raw['test'].astype('float32')/255;self.labels=raw['test_labels'];self.ids=np.arange(len(self.clean))+50000
            else:
                ids=np.arange(len(raw['train_labels']))
                tr,va=train_test_split(ids,test_size=.1,stratify=raw['train_labels'],random_state=2026)
                self.ids=tr if split=='train' else va
                self.clean=raw['train'][self.ids].astype('float32')/255;self.labels=raw['train_labels'][self.ids]
            self.parents=self.ids;self.shape=(3,32,32)
    def __len__(self):return len(self.ids)
    def __getitem__(self,j):
        ident=int(self.ids[j]);parent=int(self.parents[j]);label=int(self.labels[j])
        # Epoch-specific training corruptions; fixed validation/test corruptions.
        epoch=self.epoch if self.split=='train' else 0
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,parent,epoch]))
        nuisance=parent%2;bias=(nuisance*2-1)*.08
        if self.motion:
            _,size,_=self.shape;phase=int(rng.integers(size));row=int(rng.integers(1,size-2));amp=float(rng.uniform(.65,1.1))
            static=rng.normal(0,.06,(1,size,size)).astype('float32')+bias
            xx=np.repeat(static[None],self.T,0)
            positions=np.floor(np.arange(self.T)*size/self.T).astype(int)
            for t,pos in enumerate(positions):xx[t,0,row:row+2,(phase+pos)%size]+=amp
            if label:xx=xx[::-1].copy()
            yy=np.int64(label);mask=np.ones_like(xx,dtype='bool')
        elif self.reconstruction:
            clean=self.clean[j];mask=rng.uniform(size=(self.T,*self.shape))>.5
            noise=rng.normal(0,.06,size=mask.shape).astype('float32')
            xx=clean[None]*mask+noise+bias;yy=clean.copy()
        else:
            clean=self.clean[j];noise=rng.normal(0,.02,size=self.shape).astype('float32')
            xx=np.repeat((clean+noise+bias)[None],self.T,0)
            mask=np.ones_like(xx,dtype='bool');yy=np.int64(label)
        return {'x':torch.from_numpy(np.asarray(xx,dtype='float32')),'y':torch.as_tensor(yy),
                'label':label,'nuisance':nuisance,'id':ident,'parent':parent,'mask':torch.from_numpy(mask)}
    def manifest(self):
        return dict(task=self.task,split=self.split,T=self.T,shape=self.shape,seed=self.seed,
                    ids=self.ids.tolist(),parents=self.parents.tolist(),labels=self.labels.tolist(),
                    corruption='deterministic from data_seed,parent,epoch; epoch=0 for val/test')

