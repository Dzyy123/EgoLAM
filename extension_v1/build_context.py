"""User-authorized full-source context track, restricted to LARY source inventory."""
import argparse,csv,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from .inventory import write,sha,norm
from abilities.prepare_ssv2 import FAMILIES

def rank(s):return hashlib.sha256(('lary-context-v1|'+s).encode()).hexdigest()
def main(a):
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);inv=Path(a.inventory);src=Path(a.ssv2);members=defaultdict(set)
 for line in (inv/'mapping.jsonl').open():
  r=json.loads(line);members[r['ssv2_id']].add(r['upstream_split'])
 ann={r['id']:r for s in ['train','validation'] for r in json.loads((src/f'{s}.json').read_text())};pilot={v['sample_id'] for v in json.load(open(a.pilot))['videos']};templates={norm(t):(family,label) for family,(pos,neg) in FAMILIES.items() for label,t in [(1,pos),(0,neg)]};groups=defaultdict(list);excluded=Counter()
 for sid,splits in members.items():
  r=ann.get(sid)
  if not r or norm(r['template']) not in templates:continue
  family,label=templates[norm(r['template'])];split='test' if 'val' in splits else 'train'
  if split=='test' and sid in pilot:excluded['pilot_exposed_test_sources']+=1;continue
  groups[split,family,label].append(sid)
 rows=[]
 for (split,family,label),ids in groups.items():
  ids=sorted(ids,key=rank);n_dev=max(1,round(len(ids)*.15)) if split=='train' else 0
  for i,sid in enumerate(ids):rows.append({'sample_id':sid,'split':'dev' if i<n_dev else split,'label':label,'family':family})
 cv2.setNumThreads(1)
 def inspect(sid):
  path=src/'20bn-something-something-v2'/f'{sid}.webm';c=cv2.VideoCapture(str(path));n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));fps=c.get(cv2.CAP_PROP_FPS);c.release()
  if n<9 or fps<=0:raise ValueError('Source video unreadable/too short: '+sid)
  indices=np.rint(np.linspace(0,n-1,9)).astype(int)
  return {'sample_id':sid,'source_video_id':'ssv2:'+sid,'video_path':str(path),'frame_indices':indices.tolist(),'frame_times_sec':(indices/fps).tolist(),'duration_sec':n/fps,'fps':fps,'num_frames':n,'lary_source_splits':sorted(members[sid])}
 ids=sorted({r['sample_id'] for r in rows},key=int)
 with ThreadPoolExecutor(max_workers=16) as pool:videos=list(pool.map(inspect,ids))
 supports={split:{f:{str(y):sum(r['split']==split and r['family']==f and r['label']==y for r in rows) for y in [0,1]} for f in FAMILIES} for split in ['train','dev','test']}
 covered=[f for f in FAMILIES if all(supports['test'][f][str(y)]>=10 and supports['train'][f][str(y)]>=20 and supports['dev'][f][str(y)]>=3 for y in [0,1])]
 # Keep all data/metrics; do not selectively discard a sparse family to improve
 # a reported aggregate. Release readiness requires all six predeclared families.
 report={'protocol':'lary-ext-context-effects-v1.0.0-rc1','n_videos':len(videos),'counts':dict(Counter(r['split'] for r in rows)),'family_label_support':supports,'families_meeting_minimum_support':covered,'release_status':'ready' if len(covered)==6 else 'coverage_limited','exclusions':dict(excluded),'primary_metric':'mean over six families of binary balanced accuracy','secondary_metrics':['pooled balanced accuracy','accuracy','macro F1','per-family confusion'],'input_policy':'Full original videos, nine uniform frames; LARY source pool only. Explicitly not original LARY crops/sample_indices.','label_scope':'Human SSv2 actual vs pretending/failure labels; not generic OSCC or robot success'}
 manifest={'format':'lary-abilities-manifest-v1','protocol_version':report['protocol'],'track':'source_context_effects','label_source_type':'published_annotation','label_source':'SSv2 human templates on the complete original video','annotation_sha256':{s:sha(src/s) for s in ['train.json','validation.json']},'inventory_mapping_sha256':sha(inv/'mapping.jsonl'),'videos':videos,'tasks':{'effect':rows},'effect_families':FAMILIES,'effect_scope':report['label_scope'],'frames_per_clip':9,'frame_policy':report['input_policy'],'primary_metric':report['primary_metric'],'release_status':report['release_status']}
 write(out/'manifest.json',manifest);write(out/'summary.json',report)
 with (out/'extraction.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['sample_id','dataset','type','perspective','video_path','has_pose','action_path','action','time','sample_indices']);writer.writeheader()
  for v in videos:writer.writerow({'sample_id':v['sample_id'],'dataset':'SSv2','type':'human','perspective':'1st&3rd','video_path':v['video_path'],'has_pose':False,'action_path':'','action':'unlabeled','time':v['duration_sec'],'sample_indices':','.join(map(str,v['frame_indices']))})
 write(out/'release_lock.json',{'protocol':report['protocol'],'manifest_sha256':sha(out/'manifest.json'),'extraction_csv_sha256':sha(out/'extraction.csv'),'separate_from':'exact-clip composition track','test_selection':'original LARY val source membership; all pilot sources excluded; no model score selection'})
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['inventory','ssv2','pilot','output']:p.add_argument('--'+k,required=True)
 main(p.parse_args())
