"""Build exact-LARY-clip compositional semantics RC1, without inspecting scores."""
import argparse,csv,json,re,hashlib
from collections import defaultdict,Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
from .inventory import write,sha

# The published whole-video action must agree with the original LARY clip action.
# Only one manually named object is allowed. This is source annotation transfer,
# not a new human annotation of the crop.
VERBS={'Pushing':'push','Pulling':'pull','Throwing':'throw','Dropping':'drop','Lifting':'lift','Opening':'open','Closing':'close','Folding':'fold','Unfolding':'unfold','Tearing':'tear','Bending':'bend','Stretching':'stretch','Twisting':'twist','Squeezing':'squeeze','Rolling':'roll','Turning':'turn','Spinning':'spin','Shaking':'shake','Holding':'hold','Moving':'move','Putting':'place','Showing':'show','Poking':'poke','Touching':'touch','Tapping':'tap','Hitting':'hit','Pulling':'pull','Pushing':'push'}
def noun(s):return re.sub(r'^(a|an|the)\s+','',' '.join(s.lower().split()))
def rank(s):return hashlib.sha256(('lary-ext-v1|'+str(s)).encode()).hexdigest()

def main(a):
 inv=Path(a.inventory);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 ontology=json.loads(Path(a.object_ontology).read_text());aliases=ontology['aliases']
 allrows=[json.loads(l) for l in (inv/'mapping.jsonl').read_text().splitlines()];counts=Counter();valid=[]
 for r in allrows:
  reason=None
  if r['reversed']:reason='reversed'
  elif not r['template'] or len(r['placeholders'])!=1:reason='no_unambiguous_single_object'
  elif r['template'].split()[0] not in VERBS:reason='template_verb_outside_locked_mapping'
  elif VERBS[r['template'].split()[0]]!=r['upstream_action']:reason='original_clip_action_differs_from_source_template'
  elif noun(r['placeholders'][0]) not in aliases:reason='unmapped_or_ambiguous_object_label'
  elif r['upstream_split']=='train' and r['source_crosses_upstream_splits']:reason='purged_train_source_also_in_upstream_val'
  elif r['upstream_split']=='val' and r['pilot_exposed']:reason='excluded_exploratory_pilot_source_from_test'
  if reason:counts[reason]+=1;continue
  valid.append({**r,'verb_name':r['upstream_action'],'noun_name':aliases[noun(r['placeholders'][0])]})
 # One deterministic eligible clip per original video, so long videos do not
 # dominate and no duplicate source can cross train/dev/test.
 bysource=defaultdict(list)
 for r in valid:bysource[r['source_video_id']].append(r)
 chosen=[min(rs,key=lambda r:rank(r['sample_id'])) for rs in bysource.values()];counts['duplicate_eligible_source_clips']=len(valid)-len(chosen)
 train=[r for r in chosen if r['upstream_split']=='train'];test=[r for r in chosen if r['upstream_split']=='val']
 nc=Counter(r['noun_name'] for r in train);vc=Counter(r['verb_name'] for r in train)
 nouns=sorted(n for n in nc if nc[n]>=20);verbs=sorted(v for v in vc if vc[v]>=40)
 eligible=[r for r in chosen if r['noun_name'] in nouns and r['verb_name'] in verbs]
 tg=defaultdict(list)
 for r in eligible:
  if r['upstream_split']=='train':tg[(r['verb_name'],r['noun_name'])].append(r)
 vc=Counter();nc=Counter();vd=Counter();nd=Counter()
 for (v,n),rs in tg.items():vc[v]+=len(rs);nc[n]+=len(rs);vd[v]+=1;nd[n]+=1
 candidates=sorted([p for p,rs in tg.items() if len(rs)>=20],key=lambda p:(-len(tg[p]),rank(p)));held=[];held_mass=0;target_mass=round(.25*sum(map(len,tg.values())));minimum_pairs=10
 for v,n in candidates:
  size=len(tg[v,n])
  if len(held)>=minimum_pairs and held_mass>=target_mass:break
  if vc[v]-size<40 or nc[n]-size<20 or vd[v]<2 or nd[n]<2:continue
  held.append((v,n));held_mass+=size;vc[v]-=size;nc[n]-=size;vd[v]-=1;nd[n]-=1
 if not held:raise ValueError('No supported held combinations')
 rows=[];selected=[]
 for pair,rs in tg.items():
  if pair in held:counts['held_pair_removed_from_probe_training']+=len(rs);continue
  rs=sorted(rs,key=lambda r:rank(r['sample_id']));n_dev=max(1,round(len(rs)*.15)) if len(rs)>=7 else 0
  for i,r in enumerate(rs):
   row={'sample_id':r['sample_id'],'split':'dev' if i<n_dev else 'train','verb':verbs.index(pair[0]),'noun':nouns.index(pair[1]),'unseen_pair':False};rows.append(row);selected.append(r)
 seen={(r['verb'],r['noun']) for r in rows if r['split']=='train'};known_v={p[0] for p in seen};known_n={p[1] for p in seen}
 for r in eligible:
  if r['upstream_split']!='val':continue
  v,n=verbs.index(r['verb_name']),nouns.index(r['noun_name'])
  if v not in known_v or n not in known_n:counts['test_unknown_component']+=1;continue
  rows.append({'sample_id':r['sample_id'],'split':'test','verb':v,'noun':n,'unseen_pair':(v,n) not in seen,'held_by_design':(r['verb_name'],r['noun_name']) in held});selected.append(r)
 def inspect(r):
  c=cv2.VideoCapture(r['video_path']);n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));fps=c.get(cv2.CAP_PROP_FPS);c.release()
  if n<1 or fps<=0 or max(r['frame_indices'])>=n:raise ValueError('Native clip/frame unavailable: '+r['sample_id'])
  return {'sample_id':r['sample_id'],'source_video_id':r['source_video_id'],'video_path':r['video_path'],'frame_indices':r['frame_indices'],'frame_times_sec':[i/fps for i in r['frame_indices']],'duration_sec':n/fps,'fps':fps,'num_frames':n,'upstream_split':r['upstream_split'],'upstream_csv_row':r['upstream_csv_row'],'seed_row':r['seed_row'],'seed_line':r['seed_line'],'source_crop_start_sec':r['start_sec']}
 cv2.setNumThreads(1)
 with ThreadPoolExecutor(max_workers=16) as pool:videos=list(pool.map(inspect,sorted(selected,key=lambda r:r['sample_id'])))
 tests=[r for r in rows if r['split']=='test'];unseen=[r for r in tests if r['unseen_pair']];unseen_pairs={(r['verb'],r['noun']) for r in unseen}
 status='ready' if len(unseen)>=200 and len(unseen_pairs)>=10 else 'coverage_limited'
 report={'protocol':'lary-ext-composition-v1.0.0-rc3','status':status,'n_videos':len(videos),'task_counts':dict(Counter(r['split'] for r in rows)),'n_verbs':len(verbs),'n_nouns':len(nouns),'held_train_pairs':len(held),'held_train_mass':held_mass,'target_train_mass':target_mass,'holdout_rule':'descending training support, hash tie-break; >=10 eligible pairs and >=25% training mass when component-support constraints allow','test_unseen_videos':len(unseen),'test_unseen_pairs':len(unseen_pairs),'exclusions':dict(counts),'selection':'train annotation counts and fixed hash only; no result-based selection','label_scope':'published LARY clip action × original SSv2 manually named single object; source/crop verb agreement required','label_limit':'noun is inherited from a single-object source, not freshly annotated on each crop; clip-level label transfer quality review remains pending','effect_status':'whole-event inheritance insufficient; no claim of formal effects evaluation','pnr_status':'missing verified PNR labels','spatial_status':'full box alignment and context controls pending'}
 manifest={'format':'lary-abilities-manifest-v1','protocol_version':report['protocol'],'label_source_type':'published_annotation','label_source':'LARY published clip action (upstream machine annotation) + SSv2 human single-object noun','upstream_commit':json.loads((inv/'summary.json').read_text())['upstream_commit'],'object_ontology_sha256':sha(a.object_ontology),'source_proof_sha256':sha(inv/'source_proof.json'),'inventory_mapping_sha256':sha(inv/'mapping.jsonl'),'frames_per_clip':9,'frame_policy':'unchanged upstream sample_indices, including repeated indices on short clips','videos':videos,'tasks':{'composition':rows},'composition':{'verbs':verbs,'nouns':nouns,'held_pairs':held,'scope':report['label_scope']},'release_status':status,'gates':{'mapping_complete':True,'source_group_disjoint':True,'clip_label_transfer_review':'pending'}}
 groups={}
 vidmap={v['sample_id']:v for v in videos}
 for row in rows:
  g=vidmap[row['sample_id']]['source_video_id']
  if g in groups and groups[g]!=row['split']:raise ValueError('Source overlap across splits')
  groups[g]=row['split']
 write(out/'manifest.json',manifest);write(out/'summary.json',report);write(out/'label_transfer_ontology.json',VERBS);write(out/'object_ontology.json',ontology)
 fields=['sample_id','dataset','type','perspective','video_path','has_pose','action_path','action','time','sample_indices']
 with (out/'extraction.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for v in videos:w.writerow({'sample_id':v['sample_id'],'dataset':'SSv2','type':'human','perspective':'1st','video_path':v['video_path'],'has_pose':False,'action_path':'','action':'unlabeled','time':v['duration_sec'],'sample_indices':','.join(map(str,v['frame_indices']))})
 write(out/'release_lock.json',{'manifest_sha256':sha(out/'manifest.json'),'extraction_csv_sha256':sha(out/'extraction.csv'),'upstream_commit':manifest['upstream_commit'],'release_status':'candidate_pending_validation','independent_test_policy':'original LARY val sources, excludes all 4,982 exploratory pilot IDs; never used for tuning'})
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--output',required=True);p.add_argument('--object-ontology',required=True);main(p.parse_args())
