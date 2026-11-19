"""Map pinned LARY SSv2 clips to original manually annotated videos.
The official seed row/line-index contract is the authority, never basename guesses.
"""
import argparse,csv,hashlib,json,re,time,importlib.util
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def blob_sha(p):
 b=Path(p).read_bytes();return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def norm(s):return ' '.join(s.lower().replace('[','').replace(']','').split())
def parse_clips(s):
 out={}
 for i,line in enumerate(str(s).replace('\\n','\n').strip().split('\n')):
  matches=re.findall(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)',line)
  if matches:
   a,b=map(float,matches[0]);duration=round(b-a,2)
   if duration>0:out[i]=(a,duration)
 return out

def main(a):
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);r=Path(a.lary_root);src=Path(a.ssv2_root);lock=json.loads(Path(a.lock).read_text());proof={}
 for item in lock['files']:
  p=r/item['path']
  if p.exists():proof[item['path']]={'git_blob_sha1':blob_sha(p),'upstream_git_blob_sha1':item['sha'],'matches':blob_sha(p)==item['sha'],'sha256':sha(p)}
 for name in ['data/seed_ssv2.csv','data/human_1st_metadata_train.csv','data/human_1st_metadata_val.csv']:
  if not proof[name]['matches']:write(out/'source_proof.json',proof);raise ValueError('Pinned upstream data differs: '+name)
 seeds=list(csv.DictReader(open(r/'data/seed_ssv2.csv')));annotations={}
 for s in ['train','validation']:
  for v in json.loads((src/f'{s}.json').read_text()):
   if v['id'] in annotations:raise ValueError('Duplicate original source ID')
   annotations[v['id']]={**v,'annotation_split':s}
 pilot_ids={v['sample_id'] for v in json.loads(Path(a.pilot).read_text())['videos']}
 rows=[];errors=[];source_splits=defaultdict(set);started=time.time()
 for split in ['train','val']:
  for line,row in enumerate(csv.DictReader(open(r/f'data/human_1st_metadata_{split}.csv'))):
   if row['dataset']!='SSv2':continue
   match=re.fullmatch(r'SSv2/(\d+)_(\d+)(_rev)?\.mp4',row['video_path'])
   if not match:errors.append({'video_path':row['video_path'],'reason':'unrecognized_official_path'});continue
   idx,li=int(match[1]),int(match[2]);seed=seeds[idx];segments=parse_clips(seed['seed_output'])
   if li not in segments:errors.append({'video_path':row['video_path'],'reason':'unparsed_seed_line'});continue
   start,duration=segments[li];sid=Path(seed['video_path']).stem;source_splits[sid].add(split);ann=annotations.get(sid)
   indices=[int(i) for i in row['sample_indices'].split(',')]
   if len(indices)!=9 or min(indices)<0:raise ValueError('Invalid upstream frame indices')
   rows.append({'sample_id':row['video_path'],'source_video_id':'ssv2:'+sid,'ssv2_id':sid,'upstream_split':split,'upstream_csv_row':line,'seed_row':idx,'seed_line':li,'reversed':bool(match[3]),'start_sec':start,'duration_sec_seed':duration,'source_duration_seed':float(seed['time']),'frame_indices':indices,'video_path':str(Path(a.clips_root)/row['video_path']),'upstream_action':row['action'],'template':ann['template'] if ann else None,'placeholders':ann.get('placeholders',[]) if ann else [],'annotation_split':ann['annotation_split'] if ann else None,'pilot_exposed':sid in pilot_ids})
 for x in rows:x['source_crosses_upstream_splits']=len(source_splits[x['ssv2_id']])>1
 # Whole-video manual labels are eligible only if the forward LARY crop retains
 # the entire source to within one native 12-FPS frame. Header checks below use
 # actual source and clipped durations, rather than trusting CSV durations alone.
 candidates=[x for x in rows if not x['reversed'] and x['template'] and x['start_sec']<=1/12+1e-6 and x['start_sec']+x['duration_sec_seed']>=x['source_duration_seed']-1/12-1e-6]
 def header(path):
  c=cv2.VideoCapture(str(path));v={'n':int(c.get(cv2.CAP_PROP_FRAME_COUNT)),'fps':c.get(cv2.CAP_PROP_FPS),'width':int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))};c.release();return v
 def inspect(x):
  y=dict(x);original=header(src/'20bn-something-something-v2'/f'{x["ssv2_id"]}.webm');clip=header(x['video_path']);reason=None
  if min(original['n'],clip['n'])<1 or min(original['fps'],clip['fps'])<=0:reason='unreadable_video'
  elif abs(original['fps']-clip['fps'])>1e-5:reason='fps_mismatch'
  elif max(x['frame_indices'])>=clip['n']:reason='upstream_frame_out_of_range'
  elif x['start_sec']>1/original['fps']+1e-6 or x['start_sec']+x['duration_sec_seed']<original['n']/original['fps']-1/original['fps']-1e-6:reason='incomplete_original_event'
  elif abs(clip['n']-original['n'])>2:reason='clip_length_not_whole_event'
  y.update(source_header=original,clip_header=clip,eligibility_error=reason);return y
 cv2.setNumThreads(1)
 with ThreadPoolExecutor(max_workers=16) as pool:checked=list(pool.map(inspect,candidates))
 eligible=[x for x in checked if not x['eligibility_error']]
 summary={'upstream_commit':lock['commit'],'original_ssv2_rows':len(rows),'mapping_errors':len(errors),'upstream_splits':dict(Counter(x['upstream_split'] for x in rows)),'reversed_rows':sum(x['reversed'] for x in rows),'unique_sources':len(source_splits),'sources_crossing_upstream_splits':sum(len(s)>1 for s in source_splits.values()),'pilot_source_overlap':len({x['ssv2_id'] for x in rows if x['pilot_exposed']}),'whole_event_candidates':len(candidates),'whole_event_verified':len(eligible),'verification_exclusions':dict(Counter(x['eligibility_error'] for x in checked if x['eligibility_error'])),'eligible_splits':dict(Counter(x['upstream_split'] for x in eligible)),'elapsed_s':time.time()-started,'label_transfer_policy':'forward whole-event crop only; <= one-frame endpoint tolerance; at most two frames length discrepancy; never infer labels for partial/reversed clips'}
 write(out/'source_proof.json',{'files':proof,'annotation_sha256':{n:sha(src/n) for n in ['train.json','validation.json']},'lock_sha256':sha(a.lock)})
 write(out/'summary.json',summary);write(out/'eligible.json',eligible);write(out/'mapping_errors.json',errors)
 with (out/'mapping.jsonl').open('w') as f:
  for row in rows:f.write(json.dumps(row)+'\n')
 with (out/'coverage_checks.jsonl').open('w') as f:
  for row in checked:f.write(json.dumps(row)+'\n')
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['lary-root','ssv2-root','clips-root','pilot','lock','output']:p.add_argument('--'+k,required=True)
 main(p.parse_args())
