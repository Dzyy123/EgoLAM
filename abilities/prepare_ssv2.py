"""Prepare REAL annotated SSv2 pilots for effects and held-out compositions.

Effects mean annotated realization versus pretending, NOT Ego4D OSCC labels.
Composition components are the action TEMPLATE and first annotated object phrase.
No labels are inferred from LAM outputs or renamed LARY subclip IDs.
"""
import argparse,csv,hashlib,json,re
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from lary_ext.core import new_dir,write_json,file_hash


FAMILIES={
 'open':('Opening something','Pretending to open something without actually opening it'),
 'close':('Closing something','Pretending to close something without actually closing it'),
 'put_in':('Putting something into something','Pretending to put something into something'),
 'put_on':('Putting something onto something','Pretending to put something onto something'),
 'take_out':('Taking something out of something','Pretending to take something out of something'),
 'tear':('Tearing something into two pieces','Pretending to be tearing something that is not tearable')}


def template(s):return ' '.join(s.replace('[','').replace(']','').lower().split())
def noun(s):return re.sub(r'^(a|an|the)\s+','', ' '.join(s.lower().split()))
def rank(s,tag):return hashlib.sha256((tag+'|'+str(s)).encode()).hexdigest()


def composition_split(train,val,verbs=8,nouns=10):
    def usable(r):
        return r.get('placeholders') and noun(r['placeholders'][0]) not in {'something','thing','object','objects','stuff'} and not any(k in template(r['template']) for k in ['pretend','failing','fail to'])
    tr=[r for r in train if usable(r)];va=[r for r in val if usable(r)]
    ns=[n for n,c in Counter(noun(r['placeholders'][0]) for r in tr).most_common(nouns)]
    vs=[v for v,c in Counter(template(r['template']) for r in tr if noun(r['placeholders'][0]) in ns).most_common(verbs)]
    def group(rows):
        out=defaultdict(list)
        for r in rows:
            pair=(template(r['template']),noun(r['placeholders'][0]))
            if pair[0] in vs and pair[1] in ns:out[pair].append(r)
        return out
    tg,vg=group(tr),group(va)
    candidates=[p for p in tg if len(tg[p])>=30 and len(vg[p])>=5]
    vc=Counter();nc=Counter()
    for p,rs in tg.items():vc[p[0]]+=len(rs);nc[p[1]]+=len(rs)
    held=[];wanted=max(1,int(.2*len(candidates)))
    for p in sorted(candidates,key=lambda p:rank(str(p),'held-combination-v1')):
        if len(held)>=wanted:break
        if vc[p[0]]-len(tg[p])<60 or nc[p[1]]-len(tg[p])<60:continue
        held.append(p);vc[p[0]]-=len(tg[p]);nc[p[1]]-=len(tg[p])
    if not held:raise ValueError('No supported composition holdout')
    rows=[]
    for pair,rs in tg.items():
        if pair in held:continue
        rs=sorted(rs,key=lambda r:rank(r['id'],'composition-v1'))
        for split,chosen in [('train',rs[:20]),('dev',rs[20:25])]:
            for r in chosen:rows.append({'sample_id':r['id'],'split':split,'verb':vs.index(pair[0]),'noun':ns.index(pair[1]),'unseen_pair':False})
    train_v={r['verb'] for r in rows if r['split']=='train'};train_n={r['noun'] for r in rows if r['split']=='train'}
    seen_pairs={(r['verb'],r['noun']) for r in rows if r['split']=='train'}
    for pair,rs in vg.items():
        vi,ni=vs.index(pair[0]),ns.index(pair[1])
        if vi not in train_v or ni not in train_n:continue
        for r in sorted(rs,key=lambda r:rank(r['id'],'composition-test-v1'))[:10]:
            rows.append({'sample_id':r['id'],'split':'test','verb':vi,'noun':ni,
                         'unseen_pair':(vi,ni) not in seen_pairs,'held_by_design':pair in held})
    return rows,{'verbs':vs,'nouns':ns,'held_pairs':held,
        'scope':'action-template × first annotated object phrase; unseen to probe training only',
        'canonicalization':'lowercase/whitespace; remove leading article only; no semantic synonym merging'}


def main(a):
    out=new_dir(a.output);source=Path(a.source)
    train=json.loads((source/'train.json').read_text());val=json.loads((source/'validation.json').read_text())
    lookup={r['id']:r for r in train+val}
    if len(lookup)!=len(train)+len(val):raise ValueError('Original train/validation IDs overlap')
    tasks={};effects=[];family_counts={}
    for family,(positive,negative) in FAMILIES.items():
        family_counts[family]={}
        for label,phrase in [(1,positive),(0,negative)]:
            tr=[r for r in train if template(r['template'])==template(phrase)]
            te=[r for r in val if template(r['template'])==template(phrase)]
            if len(tr)<50 or len(te)<10:raise ValueError('Insufficient annotated family: '+phrase)
            tr=sorted(tr,key=lambda r:rank(r['id'],'effect-v1'));te=sorted(te,key=lambda r:rank(r['id'],'effect-test-v1'))
            for split,selected in [('train',tr[:200]),('dev',tr[200:240]),('test',te[:80])]:
                family_counts[family][f'{label}/{split}']=len(selected)
                for r in selected:effects.append({'sample_id':r['id'],'split':split,'label':label,'family':family})
    tasks['effect']=effects
    tasks['composition'],composition_info=composition_split(train,val)
    ids=sorted({r['sample_id'] for rows in tasks.values() for r in rows},key=lambda x:int(x))
    video_root=source/'20bn-something-something-v2'
    available={p.stem for p in video_root.iterdir() if p.suffix=='.webm'}
    missing=[i for i in ids if i not in available]
    if missing:write_json(out/'missing_videos.json',missing);raise FileNotFoundError(f'{len(missing)} raw videos missing')
    def inspect(i):
        cv2.setNumThreads(1);p=video_root/(i+'.webm');cap=cv2.VideoCapture(str(p))
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));cap.release()
        if n<9 or not np.isfinite(fps) or fps<=0:raise ValueError('Invalid video metadata: '+i)
        indices=np.rint(np.linspace(0,n-1,9)).astype(int)
        return {'sample_id':i,'source_video_id':'ssv2:'+i,'video_path':str(p),
            'frame_indices':indices.tolist(),'frame_times_sec':(indices/fps).tolist(),
            'duration_sec':n/fps,'fps':fps,'num_frames':n}
    with ThreadPoolExecutor(max_workers=16) as pool:videos=list(pool.map(inspect,ids))
    with (out/'extraction.csv').open('w',newline='') as f:
        fields=['sample_id','dataset','type','perspective','video_path','has_pose','action_path','action','time','sample_indices']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for v in videos:w.writerow({'sample_id':v['sample_id'],'dataset':'SSv2','type':'human','perspective':'1st',
            'video_path':v['video_path'],'has_pose':False,'action_path':'','action':'unlabeled',
            'time':v['duration_sec'],'sample_indices':','.join(map(str,v['frame_indices']))})
    write_json(out/'manifest.json',{'format':'lary-abilities-manifest-v1','label_source':'existing SSv2 human annotations',
        'label_source_type':'published_annotation','annotation_sha256':{n:file_hash(source/n) for n in ['train.json','validation.json','labels.json']},
        'frames_per_clip':9,'videos':videos,'tasks':tasks,'composition':composition_info,'effect_families':FAMILIES,
        'effect_scope':'annotated realization vs pretending; NOT Ego4D OSCC or physical success',
        'spatial_status':'blocked_pending_object_annotations','pnr_status':'blocked_pending_time_annotations'})
    write_json(out/'summary.json',{'n_videos':len(videos),'task_counts':{k:dict(Counter(r['split'] for r in v)) for k,v in tasks.items()},
        'effect_families':family_counts,'held_combinations':composition_info['held_pairs'],
        'source':'raw SSv2 IDs with original manual template/placeholder labels; no guessed LARY subclip mapping'})
    print((out/'summary.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);main(p.parse_args())
