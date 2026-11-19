"""Source membership, human labels, group isolation and native binary-head smoke."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from abilities.train import validate_rows
from .inventory import sha,write,norm
from .native import Probe,optimizer
from .train_context import effect_metrics

def main(a):
 W=Path(a.work);data=W/'context_effects_rc1';m=json.load(open(data/'manifest.json'));rows=validate_rows(m,'effect');ann={r['id']:r for split in ['train','validation'] for r in json.load(open(Path(a.ssv2)/f'{split}.json'))};source=defaultdict(set)
 for line in open(Path(a.inventory)/'mapping.jsonl'):
  x=json.loads(line);source[x['ssv2_id']].add(x['upstream_split'])
 assert sha(Path(a.inventory)/'mapping.jsonl')==m['inventory_mapping_sha256']
 pilot={v['sample_id'] for v in json.load(open(a.pilot))['videos']};groups=defaultdict(set)
 for r in rows:
  sid=r['sample_id'];assert sid in source;groups[r['split']].add(sid);expected=m['effect_families'][r['family']][0 if r['label']==1 else 1];assert norm(ann[sid]['template'])==norm(expected)
  if r['split']=='test':assert 'val' in source[sid] and sid not in pilot
  else:assert source[sid]=={'train'}
 assert not (groups['train']&groups['dev'] or groups['train']&groups['test'] or groups['dev']&groups['test'])
 for v in m['videos']:
  assert v['frame_indices']==np.rint(np.linspace(0,v['num_frames']-1,9)).astype(int).tolist()
  assert Path(v['video_path']).resolve()==(Path(a.ssv2)/'20bn-something-something-v2'/(v['sample_id']+'.webm')).resolve()
 for s in ['train.json','validation.json']:assert sha(Path(a.ssv2)/s)==m['annotation_sha256'][s]
 fixture=[{'label':0,'family':'A'},{'label':1,'family':'A'},{'label':1,'family':'A'},{'label':0,'family':'B'},{'label':0,'family':'B'},{'label':1,'family':'B'}]
 assert effect_metrics(np.ones(6,int),fixture)['family_macro_balanced_accuracy']==.5
 assert effect_metrics(np.array([r['label'] for r in fixture]),fixture)['family_macro_balanced_accuracy']==1
 torch.set_num_threads(4);torch.manual_seed(0)
 for dim,tokens in [(32,1),(768,16)]:
  net=Probe(a.lary_root,dim,'effect').cuda();x=torch.randn(64,8,tokens,dim,device='cuda');y={'label':torch.arange(64,device='cuda')%2};opt,ls,ws=optimizer(net,.0003,.8,20,a.native_eval)
  for _ in range(2):
   ls.step();ws.step();opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):p=net(x);loss=net.loss(p,y)
   assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(t.grad).all() for t in net.parameters() if t.grad is not None);opt.step()
  del net,opt,x,p,loss;torch.cuda.empty_cache()
 report={'protocol_version':m['protocol_version'],'manifest_sha256':sha(data/'manifest.json'),'technical_checks_passed':True,'checks':['all sources are in pinned LARY inventory','human SSv2 label identity and annotation hashes','source-group split isolation','all previous pilot sources excluded from test','whole-source uniform frame indices','family-balanced metric known answers','both latent formats pass native binary-head bf16 batch64 backward'],'independent_integrity_review':'pending','upstream_endorsement':False}
 write(W/'dataset_validation.json',report);write(W/'training_smoke.json',{'passed':True,'scope':'native binary head and metric smoke only'});print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['work','ssv2','inventory','pilot','lary-root','native-eval']:p.add_argument('--'+k,required=True)
 main(p.parse_args())
