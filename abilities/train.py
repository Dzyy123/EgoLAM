"""Frozen LAM -> native LARY attentive probe -> task metric.

Original projection, temporal embedding and 4-block pooler are reused. Only
task outputs/losses differ. Independent heads per task; selection on dev only.
"""
import argparse,csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from lary_ext.core import new_dir,write_json,file_hash
from .models import AbilityProbe
from . import metrics


GRID=[(.005,.01),(.003,.1),(.001,.4),(.0003,.8),(.0001,.01)]


def validate_rows(manifest,task):
    rows=manifest['tasks'].get(task)
    if not rows:raise ValueError('No verified annotations for task '+task)
    vids={v['sample_id']:v for v in manifest['videos']}
    groups={};seen_ids=set()
    for r in rows:
        if r['sample_id'] not in vids:raise ValueError('Unknown video sample')
        if r['sample_id'] in seen_ids:raise ValueError('Duplicate task sample')
        seen_ids.add(r['sample_id'])
        if r['split'] not in ['train','dev','test']:raise ValueError('Invalid split')
        group=vids[r['sample_id']]['source_video_id']
        if group in groups and groups[group]!=r['split']:raise ValueError('Source video crosses task splits')
        groups[group]=r['split']
        if task=='effect' and r.get('label') not in (0,1):raise ValueError('Missing real effect label')
        if task=='pnr' and 'pnr_sec' not in r:raise ValueError('Missing PNR label; midpoint is not ground truth')
        if task=='spatial' and ('box' not in r or 'context_path' not in r):raise ValueError('Spatial annotations/context missing')
    if any(not any(r['split']==s for r in rows) for s in ['train','dev','test']):raise ValueError('Empty task split')
    if task=='composition':
        seen={(r['verb'],r['noun']) for r in rows if r['split']=='train'}
        vv={p[0] for p in seen};nn={p[1] for p in seen}
        for r in rows:
            if r['verb'] not in vv or r['noun'] not in nn:raise ValueError('Unseen component is not held-out-combination evaluation')
            if r['split']=='dev' and (r['verb'],r['noun']) not in seen:raise ValueError('Development split must not tune on held combinations')
            if r['split']=='test' and bool(r['unseen_pair'])!=((r['verb'],r['noun']) not in seen):raise ValueError('Incorrect unseen-pair flag')
    return rows


def pack(manifest_path,latent_csv,latent_root,output):
    manifest=json.loads(Path(manifest_path).read_text())
    with open(latent_csv,newline='') as f:source=list(csv.DictReader(f))
    by_id={str(r['sample_id']):r for r in source}
    if len(by_id)!=len(source):raise ValueError('Duplicate extracted sample IDs')
    videos=manifest['videos'];out=new_dir(output);array=None;shape=None
    for i,v in enumerate(videos):
        r=by_id.get(v['sample_id'])
        if not r or not r.get('la_path'):raise ValueError('Missing extraction: '+v['sample_id'])
        if Path(r['video_path']).resolve()!=Path(v['video_path']).resolve():raise ValueError('Video identity mismatch')
        expected=','.join(map(str,v['frame_indices']))
        if r['sample_indices']!=expected:raise ValueError('Sampling differs across models')
        p=Path(r['la_path']);p=p if p.is_absolute() else Path(latent_root)/p
        with np.load(p,allow_pickle=False) as data:x=np.asarray(data['tokens'],np.float32)
        if x.ndim!=3 or x.shape[0]!=8 or not np.isfinite(x).all():raise ValueError('Expected finite [8,K,D] tokens')
        if array is None:
            shape=x.shape;array=np.lib.format.open_memmap(out/'tokens.npy',mode='w+',dtype='float32',shape=(len(videos),*shape))
        if x.shape!=shape:raise ValueError('Inconsistent clip shape')
        array[i]=x
    array.flush();np.save(out/'ids.npy',np.array([v['sample_id'] for v in videos]),allow_pickle=False)
    write_json(out/'manifest.json',{'format':'lary-ability-cache-v1','task_manifest_sha256':file_hash(manifest_path),
        'extraction_csv_sha256':file_hash(latent_csv),'tokens_sha256':file_hash(out/'tokens.npy'),
        'shape':[len(videos),*shape],'sampler':'nine common uniformly spaced frames; native LARY video preprocessing'})


def evaluate(net,x,rows,ids,manifest,device,batch_size,context=None,xy=None):
    preds=[];net.eval()
    index={k:i for i,k in enumerate(ids)}
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size];ix=torch.tensor([index[r['sample_id']] for r in batch],device=device)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
                o=net(x.index_select(0,ix),context.index_select(0,ix) if context is not None else None,
                      xy.index_select(0,ix) if xy is not None else None)
            preds.append(o.float().cpu().numpy())
    p=np.concatenate(preds)
    if net.task=='effect':result=metrics.classification(p.argmax(1),[r['label'] for r in rows],2);selection=result['balanced_accuracy']
    elif net.task=='composition':
        result=metrics.composition(p[:,:net.nverbs].argmax(1),p[:,net.nverbs:].argmax(1),
            [r['verb'] for r in rows],[r['noun'] for r in rows],[r.get('unseen_pair',False) for r in rows],net.nverbs,net.nnouns)
        selection=result['all']['accuracy']
    elif net.task=='pnr':
        vs={v['sample_id']:v for v in manifest['videos']};edges=np.array([vs[r['sample_id']]['frame_times_sec'] for r in rows])
        b=p[:,:net.steps].argmax(1);ii=np.arange(len(b));mid=(edges[ii,b]+edges[ii,b+1])/2
        time_pred=mid+np.tanh(p[:,net.steps:][ii,b])*(edges[ii,b+1]-edges[ii,b])/2
        result=metrics.temporal(time_pred,[r['pnr_sec'] for r in rows],edges[:,-1]-edges[:,0]);selection=-result['mae_seconds']
        result['midpoint_prior']=metrics.temporal((edges[:,0]+edges[:,-1])/2,[r['pnr_sec'] for r in rows],edges[:,-1]-edges[:,0])
    else:result=metrics.localization(p,[r['box'] for r in rows]);selection=result['mean_iou']
    return result,selection,p


def train(a):
    torch.set_num_threads(4)
    manifest=json.loads(Path(a.manifest).read_text());rows=validate_rows(manifest,a.task)
    if manifest.get('label_source_type') not in ('published_annotation','verified_manual_annotation','simulator_ground_truth'):
        raise ValueError('Formal training requires real annotation provenance; synthetic labels are smoke-only')
    cache=Path(a.cache);cm=json.loads((cache/'manifest.json').read_text())
    if cm['task_manifest_sha256']!=file_hash(a.manifest):raise ValueError('Manifest changed since extraction/packing')
    if file_hash(cache/'tokens.npy')!=cm['tokens_sha256']:raise ValueError('Feature cache changed')
    all_ids=np.load(cache/'ids.npy',allow_pickle=False).tolist();all_index={k:i for i,k in enumerate(all_ids)}
    wanted={r['sample_id'] for r in rows};ids=[i for i in all_ids if i in wanted]
    index={k:i for i,k in enumerate(ids)}
    raw=np.load(cache/'tokens.npy',mmap_mode='r',allow_pickle=False)
    x=torch.from_numpy(np.array(raw[[all_index[k] for k in ids]],copy=True)).to(a.device)
    tr=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev'];test=[r for r in rows if r['split']=='test']
    ci=manifest.get('composition',{});nv=len(ci.get('verbs',[]));nn=len(ci.get('nouns',[]))
    context=xy=None
    if a.task=='spatial':
        contexts={};positions={}
        for r in rows:
            with np.load(r['context_path'],allow_pickle=False) as f:
                contexts[r['sample_id']]=f['tokens'];positions[r['sample_id']]=f['xy']
        # A spatial manifest must contain only the corresponding common cohort.
        if set(contexts)!=set(ids):raise ValueError('Shared context coverage differs from latent cache')
        context=torch.from_numpy(np.stack([contexts[k] for k in ids])).float().to(a.device)
        xy=torch.from_numpy(np.stack([positions[k] for k in ids])).float().to(a.device)
    out=new_dir(a.output);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    grid=GRID if a.grid=='native' else [(float(a.lr),float(a.weight_decay))]
    nets=[];opts=[];schedulers=[];best=[-float('inf')]*len(grid);states=[None]*len(grid);epochs=[0]*len(grid)
    total_steps=a.epochs*math.ceil(len(tr)/a.batch_size)
    for lr,wd in grid:
        net=AbilityProbe(a.lary_root,x.shape[-1],a.task,nverbs=nv,nnouns=nn,context_dim=context.shape[-1] if context is not None else None,
            condition=a.condition).to(a.device)
        opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=wd)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:.5*(1+math.cos(math.pi*min(step,total_steps)/total_steps)))
        nets.append(net);opts.append(opt);schedulers.append(scheduler)
    videos={v['sample_id']:v for v in manifest['videos']};history=[];started=time.time()
    for epoch in range(a.epochs):
        for n in nets:n.train()
        order=rng.permutation(len(tr));losses=[[] for _ in nets]
        for start in range(0,len(tr),a.batch_size):
            batch=[tr[i] for i in order[start:start+a.batch_size]]
            ix=torch.tensor([index[r['sample_id']] for r in batch],device=a.device)
            target={}
            if a.task=='effect':target['label']=torch.tensor([r['label'] for r in batch],device=a.device)
            elif a.task=='composition':
                for k in ['verb','noun']:target[k]=torch.tensor([r[k] for r in batch],device=a.device)
            elif a.task=='pnr':
                bins,offset=metrics.pnr_targets([videos[r['sample_id']]['frame_times_sec'] for r in batch],[r['pnr_sec'] for r in batch])
                target={'bin':torch.tensor(bins,device=a.device),'offset':torch.tensor(offset,device=a.device)}
            else:target['box']=torch.tensor([r['box'] for r in batch],device=a.device)
            for h,(net,opt,sched) in enumerate(zip(nets,opts,schedulers)):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(a.device).startswith('cuda')):
                    p=net(x.index_select(0,ix),context.index_select(0,ix) if context is not None else None,xy.index_select(0,ix) if xy is not None else None)
                    loss=net.loss(p,target)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();opt.step();sched.step();losses[h].append(float(loss.detach()))
        scores=[]
        for h,net in enumerate(nets):
            result,value,_=evaluate(net,x,dev,ids,manifest,a.device,a.batch_size,context,xy);scores.append(value)
            if value>best[h]:
                best[h]=value;epochs[h]=epoch+1;states[h]={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        record={'epoch':epoch+1,'train_losses':[float(np.mean(v)) for v in losses],'development_scores':scores,'elapsed_s':time.time()-started}
        history.append(record);write_json(out/'progress.json',record);print(json.dumps(record),flush=True)
    chosen=int(np.argmax(best));nets[chosen].load_state_dict(states[chosen])
    result,_,pred=evaluate(nets[chosen],x,test,ids,manifest,a.device,a.batch_size,context,xy)
    torch.save({'state_dict':states[chosen],'task':a.task,'input_dim':x.shape[-1],'model_dim':768,'depth':4,'heads':16},out/'probe.pth')
    np.savez(out/'predictions.npz',logits_or_boxes=pred,sample_ids=np.array([r['sample_id'] for r in test]))
    write_json(out/'result.json',{'task':a.task,'test':result,'selected_grid':grid[chosen],'selected_epoch':epochs[chosen],
        'selection':'development only; test used once','all_development_best':best,'grid':grid,
        'n_train':len(tr),'n_dev':len(dev),'n_test':len(test),'manifest_sha256':file_hash(a.manifest),
        'cache_manifest_sha256':file_hash(cache/'manifest.json'),'native_pooler_sha256':file_hash(Path(a.lary_root)/'classification/src/models/attentive_pooler.py'),
        'scope':manifest.get('effect_scope') if a.task=='effect' else ci.get('scope'),'history':history,
        'model_parameters':sum(p.numel() for p in nets[chosen].parameters()),'condition':a.condition,
        'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda},'seed':a.seed,
        'formal_original_LARY_score':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('pack')
    for k in ['manifest','latent-csv','latent-root','output']:q.add_argument('--'+k,required=True)
    q=sub.add_parser('train')
    for k in ['lary-root','manifest','cache','output']:q.add_argument('--'+k,required=True)
    q.add_argument('--task',choices=['effect','composition','pnr','spatial'],required=True)
    q.add_argument('--epochs',type=int,default=20);q.add_argument('--batch-size',type=int,default=64)
    q.add_argument('--device',default='cuda:0');q.add_argument('--seed',type=int,default=0)
    q.add_argument('--grid',choices=['native','single'],default='native');q.add_argument('--lr',type=float,default=.0003);q.add_argument('--weight-decay',type=float,default=.01)
    q.add_argument('--condition',choices=['full','context_only'],default='full')
    a=p.parse_args()
    if a.command=='pack':pack(a.manifest,a.latent_csv,a.latent_root,a.output)
    else:train(a)
