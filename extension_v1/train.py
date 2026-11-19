"""Native LARY probe protocol for a versioned composition manifest.
Training selects on development data only. Held-out test scoring is separate.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DistributedSampler
from abilities.train import validate_rows
from abilities.metrics import composition
from .native import Probe,GRID,optimizer
from .inventory import sha,write,blob_sha


def setup(a):
 torch.set_num_threads(4);torch.manual_seed(a.seed);torch.backends.cudnn.benchmark=True
 m=json.loads(Path(a.manifest).read_text());rows=validate_rows(m,'composition');cache=Path(a.cache);cm=json.loads((cache/'manifest.json').read_text())
 if cm['task_manifest_sha256']!=sha(a.manifest):raise ValueError('Manifest/cache mismatch')
 if cm['tokens_sha256']!=sha(cache/'tokens.npy'):raise ValueError('Cache contents changed')
 pooler=Path(a.lary_root)/'classification/src/models/attentive_pooler.py'
 if blob_sha(pooler)!='7765143ec07d3da8500af2460f5f9df891f867f6':raise ValueError('Native probe differs from pinned LARY version')
 ids=np.load(cache/'ids.npy',allow_pickle=False).tolist();idx={s:i for i,s in enumerate(ids)};x=torch.from_numpy(np.load(cache/'tokens.npy',allow_pickle=False)).to('cuda')
 return m,rows,x,idx


def evaluate(net,x,rows,idx,batch=64):
 net.eval();pred=[]
 with torch.inference_mode():
  for start in range(0,len(rows),batch):
   ix=torch.tensor([idx[r['sample_id']] for r in rows[start:start+batch]],device='cuda')
   # Native run_epoch enables bf16 only while training; evaluation is FP32.
   logits=net(x.index_select(0,ix))
   if not torch.isfinite(logits).all():raise FloatingPointError('Non-finite evaluation output')
   pred.append(logits.cpu().numpy())
 p=np.concatenate(pred);result=composition(p[:,:net.nverbs].argmax(1),p[:,net.nverbs:].argmax(1),[r['verb'] for r in rows],[r['noun'] for r in rows],[r.get('unseen_pair',False) for r in rows],net.nverbs,net.nnouns)
 return result,p


def train(a):
 m,rows,x,idx=setup(a);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 tr=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];nv=len(m['composition']['verbs']);nn=len(m['composition']['nouns']);total=a.epochs*math.ceil(len(tr)/64)
 native_eval=Path(a.native_eval);nets=[];opts=[];schedules=[];best=[-float('inf')]*5;states=[None]*5;epochs=[None]*5;history=[];start_time=time.time()
 for lr,wd in GRID:
  net=Probe(a.lary_root,x.shape[-1],'composition',nv,nn).cuda();opt,ls,ws=optimizer(net,lr,wd,total,native_eval);nets.append(net);opts.append(opt);schedules.append((ls,ws))
 sampler=DistributedSampler(tr,num_replicas=1,rank=0,shuffle=True,seed=0,drop_last=False)
 for epoch in range(a.epochs):
  sampler.set_epoch(epoch);order=list(sampler);losses=[[] for _ in nets]
  for n in nets:n.train()
  for start in range(0,len(tr),64):
   batch=[tr[i] for i in order[start:start+64]];ix=torch.tensor([idx[r['sample_id']] for r in batch],device='cuda');y={k:torch.tensor([r[k] for r in batch],device='cuda') for k in ['verb','noun']}
   for j,(net,opt,ss) in enumerate(zip(nets,opts,schedules)):
    # Native schedules update before the first optimizer update.
    for sched in ss:sched.step()
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):p=net(x.index_select(0,ix));loss=net.loss(p,y)
    if not torch.isfinite(loss) or not torch.isfinite(p).all():raise FloatingPointError(f'Non-finite head {j}; stopping instead of silently dropping it')
    loss.backward();opt.step();opt.zero_grad(set_to_none=True);losses[j].append(float(loss.detach()))
  values=[]
  for j,net in enumerate(nets):
   result,_=evaluate(net,x,dev,idx);score=result['all']['accuracy'];values.append(score)
   if score>best[j]:best[j]=score;states[j]={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};epochs[j]=epoch+1
  record={'epoch':epoch+1,'train_loss':[float(np.mean(v)) for v in losses],'development_joint_accuracy':values,'elapsed_seconds':time.time()-start_time};history.append(record);write(out/'progress.json',record);print(json.dumps(record),flush=True)
 selected=int(np.argmax(best));torch.save({'state_dict':states[selected],'input_dim':x.shape[-1],'nverbs':nv,'nnouns':nn,'seed':a.seed,'manifest_sha256':sha(a.manifest)},out/'probe.pt')
 write(out/'selection.json',{'protocol_version':m['protocol_version'],'seed':a.seed,'manifest_sha256':sha(a.manifest),'cache_manifest_sha256':sha(Path(a.cache)/'manifest.json'),'selected_grid':GRID[selected],'selected_epoch':epochs[selected],'development_best':best,'selection':'development joint accuracy; no test predictions during training','history':history,'native_eval_sha256':sha(native_eval),'native_pooler_sha256':sha(Path(a.lary_root)/'classification/src/models/attentive_pooler.py'),'label_smoothing':.1,'train_precision':'bf16','evaluation_precision':'fp32','batch_size':64,'epochs':a.epochs,'sampler':'native single-rank DistributedSampler, seed 0, set_epoch','checkpoint_sha256':sha(out/'probe.pt'),'torch':torch.__version__,'cuda':torch.version.cuda})


def score(a):
 m,rows,x,idx=setup(a);report=json.loads(Path(a.validation).read_text())
 if report.get('manifest_sha256')!=sha(a.manifest) or not report.get('technical_checks_passed'):raise ValueError('Dataset validation missing or wrong manifest')
 out=Path(a.output);selection=json.loads((out/'selection.json').read_text())
 if sha(out/'probe.pt')!=selection['checkpoint_sha256']:raise ValueError('Changed probe checkpoint')
 if (out/'result.json').exists():raise FileExistsError('Held-out score already exists')
 ck=torch.load(out/'probe.pt',map_location='cpu',weights_only=True)
 if ck['manifest_sha256']!=sha(a.manifest):raise ValueError('Checkpoint/manifest mismatch')
 net=Probe(a.lary_root,ck['input_dim'],'composition',ck['nverbs'],ck['nnouns']).cuda();net.load_state_dict(ck['state_dict'],strict=True)
 test=[r for r in rows if r['split']=='test'];result,p=evaluate(net,x,test,idx)
 np.savez(out/'predictions.npz',logits=p,sample_ids=np.array([r['sample_id'] for r in test]))
 write(out/'result.json',{'protocol_version':m['protocol_version'],'seed':a.seed,'test':result,'selection_sha256':sha(out/'selection.json'),'manifest_sha256':sha(a.manifest),'dataset_validation_sha256':sha(a.validation),'scope':m['composition']['scope'],'upstream_official_result':False,'independent_integrity_review':'pending'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','score'])
 for k in ['lary-root','manifest','cache','output']:p.add_argument('--'+k,required=True)
 p.add_argument('--native-eval');p.add_argument('--validation');p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=20);a=p.parse_args();train(a) if a.mode=='train' else score(a)
