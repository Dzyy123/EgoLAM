"""Full-source effect realization with native LARY probe and family-balanced scoring."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DistributedSampler
from abilities.train import validate_rows
from abilities.metrics import classification
from .native import Probe,GRID,optimizer
from .inventory import sha,write,blob_sha

def effect_metrics(pred,rows):
 y=np.array([r['label'] for r in rows]);p=np.asarray(pred);pooled=classification(p,y,2);families={}
 for family in sorted({r['family'] for r in rows}):
  ix=np.array([r['family']==family for r in rows]);assert set(y[ix])=={0,1},'Both labels required in every family'
  families[family]=classification(p[ix],y[ix],2)
 return {'family_macro_balanced_accuracy':float(np.mean([v['balanced_accuracy'] for v in families.values()])),'pooled':pooled,'families':families}

def setup(a):
 torch.set_num_threads(4);torch.manual_seed(a.seed);torch.backends.cudnn.benchmark=True;m=json.load(open(a.manifest));rows=validate_rows(m,'effect');cache=Path(a.cache);cm=json.load(open(cache/'manifest.json'))
 assert cm['task_manifest_sha256']==sha(a.manifest) and cm['tokens_sha256']==sha(cache/'tokens.npy')
 assert blob_sha(Path(a.lary_root)/'classification/src/models/attentive_pooler.py')=='7765143ec07d3da8500af2460f5f9df891f867f6'
 ids=np.load(cache/'ids.npy',allow_pickle=False).tolist();idx={s:i for i,s in enumerate(ids)};x=torch.from_numpy(np.load(cache/'tokens.npy',allow_pickle=False)).cuda();return m,rows,x,idx

def evaluate(net,x,rows,idx):
 net.eval();pred=[]
 with torch.inference_mode():
  for start in range(0,len(rows),64):
   ix=torch.tensor([idx[r['sample_id']] for r in rows[start:start+64]],device='cuda');p=net(x.index_select(0,ix));assert torch.isfinite(p).all();pred.append(p.cpu().numpy())
 p=np.concatenate(pred);return effect_metrics(p.argmax(1),rows),p

def train(a):
 m,rows,x,idx=setup(a);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);tr=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];steps=a.epochs*math.ceil(len(tr)/64);nets=[];opts=[];schedules=[];best=[-1.]*5;states=[None]*5;epochs=[0]*5;history=[];started=time.time()
 for lr,wd in GRID:
  net=Probe(a.lary_root,x.shape[-1],'effect').cuda();opt,ls,ws=optimizer(net,lr,wd,steps,a.native_eval);nets.append(net);opts.append(opt);schedules.append((ls,ws))
 sampler=DistributedSampler(tr,num_replicas=1,rank=0,shuffle=True,seed=0,drop_last=False)
 for epoch in range(a.epochs):
  sampler.set_epoch(epoch);order=list(sampler);losses=[[] for _ in nets]
  for net in nets:net.train()
  for start in range(0,len(tr),64):
   batch=[tr[i] for i in order[start:start+64]];ix=torch.tensor([idx[r['sample_id']] for r in batch],device='cuda');y={'label':torch.tensor([r['label'] for r in batch],device='cuda')}
   for h,(net,opt,ss) in enumerate(zip(nets,opts,schedules)):
    for sched in ss:sched.step()
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):p=net(x.index_select(0,ix));loss=net.loss(p,y)
    if not torch.isfinite(loss) or not torch.isfinite(p).all():raise FloatingPointError('Nonfinite head; stopping whole comparison')
    loss.backward();opt.step();losses[h].append(float(loss.detach()))
  values=[]
  for h,net in enumerate(nets):
   result,_=evaluate(net,x,dev,idx);v=result['family_macro_balanced_accuracy'];values.append(v)
   if v>best[h]:best[h]=v;epochs[h]=epoch+1;states[h]={k:t.detach().cpu().clone() for k,t in net.state_dict().items()}
  record={'epoch':epoch+1,'train_loss':[float(np.mean(v)) for v in losses],'development_family_macro_balanced_accuracy':values,'elapsed_seconds':time.time()-started};history.append(record);write(out/'progress.json',record);print(json.dumps(record),flush=True)
 chosen=int(np.argmax(best));torch.save({'state_dict':states[chosen],'input_dim':x.shape[-1],'seed':a.seed,'manifest_sha256':sha(a.manifest)},out/'probe.pt')
 write(out/'selection.json',{'protocol_version':m['protocol_version'],'seed':a.seed,'manifest_sha256':sha(a.manifest),'cache_manifest_sha256':sha(Path(a.cache)/'manifest.json'),'selected_grid':GRID[chosen],'selected_epoch':epochs[chosen],'development_best':best,'selection':'development mean family balanced accuracy; no test scoring in training','history':history,'label_smoothing':.1,'train_precision':'bf16','evaluation_precision':'fp32','batch_size':64,'epochs':a.epochs,'checkpoint_sha256':sha(out/'probe.pt'),'torch':torch.__version__,'cuda':torch.version.cuda})

def score(a):
 m,rows,x,idx=setup(a);validation=json.load(open(a.validation));assert validation['technical_checks_passed'] and validation['manifest_sha256']==sha(a.manifest);out=Path(a.output);selection=json.load(open(out/'selection.json'));assert sha(out/'probe.pt')==selection['checkpoint_sha256']
 if (out/'result.json').exists():raise FileExistsError('Test already scored')
 ck=torch.load(out/'probe.pt',map_location='cpu',weights_only=True);assert ck['manifest_sha256']==sha(a.manifest);net=Probe(a.lary_root,ck['input_dim'],'effect').cuda();net.load_state_dict(ck['state_dict'],strict=True);test=[r for r in rows if r['split']=='test'];result,p=evaluate(net,x,test,idx)
 np.savez(out/'predictions.npz',logits=p,sample_ids=np.array([r['sample_id'] for r in test]));write(out/'result.json',{'protocol_version':m['protocol_version'],'seed':a.seed,'test':result,'manifest_sha256':sha(a.manifest),'selection_sha256':sha(out/'selection.json'),'scope':m['effect_scope'],'upstream_official_result':False,'independent_integrity_review':'pending'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','score'])
 for k in ['lary-root','manifest','cache','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--native-eval');p.add_argument('--validation');p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=20);a=p.parse_args();train(a) if a.mode=='train' else score(a)
