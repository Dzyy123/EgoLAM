import csv, hashlib, json, math, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

import argparse
parser=argparse.ArgumentParser(description="Match measured wrist motion within videos during large and small head turns")
for name in ("cohort","pairs-root","readout","output"):
    parser.add_argument("--"+name,type=Path,required=True)
parser.add_argument("--split",choices=["unseen","val"],required=True)
args=parser.parse_args()
split=args.split
inp=args.cohort
out=args.output
out.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name): return json.loads((inp/name).read_text())
def csv_rows(name):
    with (inp/name).open() as f: return list(csv.DictReader(f))
def write(name,x): (out/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def values(rows,cols): return np.array([[float(r[c] or "nan") for c in cols] for r in rows],dtype=np.float64)
def key(r): return r['subset'], r['episode_id']
def flag(v):return v.strip().lower() in ('true','1')
def tensor(x):return torch.as_tensor(x,device='cuda',dtype=torch.float64)
def quat(v):
    a=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    return torch.cat([torch.cos(a/2),0.5*torch.sinc(a/(2*math.pi))*v],dim=-1)
def qdist(q,r):
    # Distance on SO(3), stable near equal rotations and invariant to quaternion sign.
    r=torch.where((q*r).sum(-1,keepdim=True)<0,-r,r)
    return 4*torch.atan2(torch.linalg.vector_norm(q-r,dim=-1),torch.linalg.vector_norm(q+r,dim=-1))*180/math.pi
def cpu(x):return x.detach().cpu().numpy()
def desc(x):
    return dict(mean=float(np.mean(x)),median=float(np.median(x)),p90=float(np.quantile(x,.90)),max=float(np.max(x)))

assert torch.cuda.is_available()
cohort=read('cohort.json')
result=json.loads((args.readout/'result.json').read_text())
assert result['model']=='EgoLAM' and result['probe']=='mlp' and result['stride']==15 and result['seed']==42
assert result['cohort_sha256']==sha(inp/'cohort.json')
assert sha(inp/f'{split}.csv')==cohort['csv_sha256'][split]
rows=csv_rows(f'{split}.csv');N=len(rows)
assert N=={'unseen':9322,'val':8904}[split]
assert [r['la_path'] for r in rows]==[f'{split}:{i}' for i in range(N)]
assert len({(*key(r),r['src_frame'],r['tgt_frame']) for r in rows})==N
A=['a_'+h+'_'+c for h in ['l','r'] for c in ['x','y','z','rx','ry','rz']]
gt=values(rows,A).astype('float32')
pp=args.readout/result['AD']['_paired_predictions'][split]
with np.load(pp) as f:
    np.testing.assert_array_equal(f['target'],gt);pred=f['prediction'].copy()
assert np.isfinite(pred).all()
y=tensor(gt); yh=tensor(pred)
r2=float((1-((y-yh)**2).mean(0)/(y.var(0,unbiased=False)+1e-12)).mean())
assert abs(r2-result['AD'][split]['all'])<1e-8
frames={}
for r in csv_rows(args.pairs_root.resolve()/f'frames_{split}.csv'):
    k=(*key(r),int(r['frame'])); assert k not in frames
    frames[k]=r
ns=values(rows,['n_'+x for x in ['tx','ty','tz','rx','ry','rz']])
poscols=['head_hx','head_hy','head_hz']
# Builder writes the seven original RLDS entries in order. Despite the legacy
# column names, the stored quaternion order is wxyz, as verified by the builder.
rotcols=['head_hqy','head_hqz','head_hqw','head_hqx']
meta=[]; exclusions=Counter();err_r=[];err_t=[]
for i,r in enumerate(rows):
    f0=int(r['src_frame']); f1=int(r['tgt_frame']);assert f1-f0==15
    x=frames[(*key(r),f0)]; z=frames[(*key(r),f1)]
    reason=None
    if r['subset']=='eva' or r['type']!='human' or r['perspective']!='1st':reason='non-human/fixed camera'
    elif not (flag(r['kp_valid_left']) and flag(r['kp_valid_right'])):reason='untracked hand'
    elif not (flag(x['has_head']) and flag(z['has_head']) and np.isfinite(ns[i]).all()):reason='missing head poses'
    else:
        q=values([x,z],rotcols);t=values([x,z],poscols)
        if not (np.isfinite(q).all() and np.isfinite(t).all() and (abs(np.linalg.norm(q,axis=1)-1)<0.05).all()):reason='invalid endpoint pose'
        else:
            rr=Rotation.from_quat(q);rel=rr[0].inv()*rr[1]
            err_r.append(np.linalg.norm(rel.as_rotvec()-ns[i,3:]));err_t.append(np.linalg.norm(rr[0].inv().apply(t[1]-t[0])-ns[i,:3]))
    if reason:exclusions[reason]+=1
    else:meta.append(i)
assert max(err_r)<1e-5 and max(err_t)<1e-5
ids=np.array(meta);groups=defaultdict(list)
for i in ids:
    r=rows[i];groups[(*key(r),r['task_name'])].append(int(i))
starts=np.array([int(r['src_frame']) for r in rows])
head_angle=np.rad2deg(np.linalg.norm(ns[:,3:],axis=1));headpos=tensor(ns[:,:3])*100
w=y.reshape(-1,2,6);wh=yh.reshape(-1,2,6);pos=100*w[:,:,:3];rot=quat(w[:,:,3:])
trans_err=torch.linalg.vector_norm(100*(wh[:,:,:3]-w[:,:,:3]),dim=2)
rot_err=qdist(quat(wh[:,:,3:]),rot)
errs=cpu(torch.cat([trans_err.mean(1,keepdim=True),rot_err.mean(1,keepdim=True),trans_err,rot_err],dim=1))
endpoints=['translation_cm','rotation_deg','left_translation_cm','right_translation_cm','left_rotation_deg','right_rotation_deg']
amp_pos=cpu(torch.linalg.vector_norm(pos,dim=-1));amp_rot=cpu(torch.linalg.vector_norm(w[:,:,3:],dim=-1))*180/math.pi
active=(amp_pos>=2)|(amp_rot>=10)
def match(hi,cm,deg,head_cap,active_only):
    matches=[]; candidates=0
    for group,ii in sorted(groups.items()):
        ii=np.array(ii);lo=ii[head_angle[ii]<=5];big=ii[head_angle[ii]>=hi]
        if not len(lo) or not len(big):continue
        dl=torch.linalg.vector_norm(pos[lo,None]-pos[big][None,:],dim=-1)
        dr=qdist(rot[lo,None],rot[big][None,:])
        dh=torch.linalg.vector_norm(headpos[lo,None]-headpos[big][None,:],dim=-1)
        valid=cpu((dl<=cm).all(-1)&(dr<=deg).all(-1))
        if head_cap:valid &= cpu(dh<=head_cap)
        valid &= abs(starts[lo,None]-starts[big][None,:])>15
        if active_only:valid &= (active[lo,None]&active[big][None,:]).any(-1)
        candidates+=int(valid.sum())
        if not valid.any():continue
        cost=((dl/cm)**2+(dr/deg)**2).sum(-1)
        if head_cap:cost+=(dh/head_cap)**2
        c=cpu(cost);penalty=10*(1+min(c.shape));I,J=linear_sum_assignment(np.where(valid,c,penalty))
        for a,b in zip(I,J):
            if valid[a,b]:
                matches.append((int(lo[a]),int(big[b]),group))
    return matches,candidates
def summarize(pairs):
    left=np.array([t[0] for t in pairs]);right=np.array([t[1] for t in pairs]);eps=[p[2][:2] for p in pairs]
    assert not set(left)&set(right) and len(set(left))==len(left) and len(set(right))==len(right)
    epnames=sorted(set(eps));emap={k:i for i,k in enumerate(epnames)};cluster=np.array([emap[k] for k in eps])
    delta=errs[right]-errs[left];counts=np.bincount(cluster);sums=np.zeros((len(epnames),6));np.add.at(sums,cluster,delta)
    rng=np.random.default_rng(20261116);boots=np.zeros((10000,6));den=np.zeros(10000)
    for s in sorted({g[0] for g in epnames}):
        ix=np.array([i for i,e in enumerate(epnames) if e[0]==s]);mult=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=len(den))
        boots+=mult@sums[ix];den+=mult@counts[ix]
    ci=np.percentile(boots/den[:,None],[2.5,97.5],axis=0)
    outcomes={k:dict(low=float(errs[left,j].mean()),high=float(errs[right,j].mean()),delta=float(delta[:,j].mean()),ci=ci[:,j].tolist()) for j,k in enumerate(endpoints)}
    dp=cpu(torch.linalg.vector_norm(pos[left]-pos[right],dim=-1));dq=cpu(qdist(rot[left],rot[right]))
    by_source={}
    for s in sorted({g[0] for g in epnames}):
        sel=np.array([g[0]==s for g in eps]);by_source[s]=dict(n=int(sel.sum()),episodes=len(set(g for g in eps if g[0]==s)),delta=delta[sel,:2].mean(0).tolist())
    return dict(n=len(pairs),episodes=len(epnames),largest_episode=int(max(counts)),by_source=by_source,
      low_head=desc(head_angle[left]),high_head=desc(head_angle[right]),balance=dict(wrist_translation_gap_cm=desc(dp.max(1)),wrist_rotation_gap_deg=desc(dq.max(1)),head_translation_gap_cm=desc(cpu(torch.linalg.vector_norm(headpos[left]-headpos[right],dim=-1))),low_wrist_motion_cm=amp_pos[left].mean(0).tolist(),high_wrist_motion_cm=amp_pos[right].mean(0).tolist(),low_wrist_motion_deg=amp_rot[left].mean(0).tolist(),high_wrist_motion_deg=amp_rot[right].mean(0).tolist()),outcomes=outcomes)

eligible_episodes=sorted(set((rows[i]['subset'],rows[i]['episode_id']) for i in ids))
def box(m):
    return {k:float(v) for k,v in zip(['mean','median','p90','maximum'],[m.mean(),np.median(m),np.percentile(m,90),max(m)])}
def contrasts(m):
    a=np.array([x[0] for x in m]);b=np.array([x[1] for x in m]);h=np.array([x[2] for x in m])
    e=sorted({x[3][:2] for x in m});emap={k:i for i,k in enumerate(e)};ci=np.array([emap[x[3][:2]] for x in m])
    low=np.stack([errs[a,2+h],errs[a,4+h]],1);high=np.stack([errs[b,2+h],errs[b,4+h]],1);diff=high-low
    count=np.bincount(ci);es=np.zeros((len(e),2));np.add.at(es,ci,diff);w=np.zeros((10000,len(e)));rng=np.random.default_rng(20261116)
    for s in sorted({g[0] for g in e}):
        ix=np.where(np.array([g[0] for g in e])==s)[0];w[:,ix]=rng.multinomial(len(ix),np.ones(len(ix))/len(ix),size=10000)
    bd=(w@es)/(w@count)[:,None]
    p,q=cpu(torch.linalg.vector_norm(pos[a,h]-pos[b,h],dim=-1)),cpu(qdist(rot[a,h],rot[b,h]))
    other=1-h
    outc=dict(n=len(a),n_episodes=len(e),max_from_one_episode=int(count.max()),source_counts=dict(Counter(rows[i]['subset'] for i in a)),
        head_rotation_median_deg=[float(np.median(head_angle[a])),float(np.median(head_angle[b]))],
        head_translation_median_cm=[float(np.median(np.linalg.norm(ns[i,:3]*100,axis=1))) for i in [a,b]],
        head_translation_gap_cm=box(cpu(torch.linalg.vector_norm(headpos[a]-headpos[b],dim=-1))),
        wrist_translation_gap_cm=box(p),wrist_rotation_gap_deg=box(q),
        true_wrist_displacement_cm=[float(amp_pos[i,h].mean()) for i in [a,b]],
        true_wrist_rotation_deg=[float(amp_rot[i,h].mean()) for i in [a,b]],
        other_hand_translation_gap_cm=box(cpu(torch.linalg.vector_norm(pos[a,other]-pos[b,other],dim=-1))),
        other_hand_rotation_gap_deg=box(cpu(qdist(rot[a,other],rot[b,other]))),
        episode_equal_delta=(es/count[:,None]).mean(0).tolist())
    for j,k in enumerate(['translation_cm','rotation_deg']):
        outc[k]=dict(low=float(low[:,j].mean()),high=float(high[:,j].mean()),delta=float(diff[:,j].mean()),ci=np.percentile(bd[:,j],[2.5,97.5]).tolist())
    return outc
def wrist_match(h,head_cap,moving):
    matches=[]
    for g,ii in sorted(groups.items()):
        ii=np.array(ii);lo=ii[head_angle[ii]<=5];hi=ii[head_angle[ii]>=15]
        if not len(lo) or not len(hi):continue
        dp=torch.linalg.vector_norm(pos[lo,None,h]-pos[hi,h][None,:],dim=-1);dq=qdist(rot[lo,None,h],rot[hi,h][None,:]);dh=torch.linalg.vector_norm(headpos[lo,None]-headpos[hi][None,:],dim=-1)
        keep=cpu((dp<=2)&(dq<=10));cost=(dp/2)**2+(dq/10)**2
        keep &= abs(starts[lo,None]-starts[hi][None,:])>15
        if head_cap: keep &= cpu(dh<=2);cost+=(dh/2)**2
        if moving:keep &= active[lo,None,h]&active[hi,h][None,:]
        if not keep.any():continue
        c=cpu(cost);u,v=linear_sum_assignment(np.where(keep,c,10*(1+min(c.shape))))
        matches.extend((int(lo[a]),int(hi[b]),h,g) for a,b in zip(u,v) if keep[a,b])
    return matches
res=dict(split=split,source_sha=sha(Path(__file__)),inputs=dict(predictions=sha(pp),cohort=sha(inp/'cohort.json'),frames=sha(args.pairs_root/f'frames_{split}.csv')),r2=r2,eligible=len(ids),episodes=len(eligible_episodes),eligible_episode_ids=['/'.join(k) for k in eligible_episodes],full_rows=N,counts=dict(Counter(rows[i]['subset'] for i in ids)),real_quaternion_error_rad=max(err_r),head_translation_check_m=max(err_t),high_n=int((head_angle[ids]>=15).sum()),low_n=int((head_angle[ids]<=5).sum()),gpu=torch.cuda.get_device_name(0))
for tag,pars in [('bimanual',(15,2,10,2,False)),('bimanual_without_head_translation',(15,2,10,None,False))]:
    mm,nedge=match(*pars);res[tag]=summarize(mm) if mm else {'n':0,'episodes':0};res[tag]['edges']=nedge
with (out/(split+'_wrist_matches.csv')).open('w') as f:
    cw=csv.writer(f);cw.writerow(['variant','hand','source','episode','task','low_index','high_index','low_src_frame','high_src_frame','low_head_deg','high_head_deg','low_cm','high_cm','low_deg','high_deg'])
    for tag,cap,act in [('per_wrist',False,False),('head_translation_matched',True,False),('active_wrist',False,True)]:
        res[tag]={};both=[]
        for h,hname in enumerate(['left','right']):
            m=wrist_match(h,cap,act);both.extend(m)
            assert len(m)==len(set(a for a,b,hh,g in m))==len(set(b for a,b,hh,g in m))
            res[tag][hname]=contrasts(m) if m else {'n':0,'n_episodes':0}
            for a,b,hh,g in m:cw.writerow([tag,hname,*g,a,b,starts[a],starts[b],head_angle[a],head_angle[b],errs[a,2+h],errs[b,2+h],errs[a,4+h],errs[b,4+h]])
        res[tag]['pooled']=contrasts(both) if both else {'n':0,'n_episodes':0}
write(split+'_secondary.json',res)
print(split,'bimanual',res['bimanual']['n'])
for t in ['per_wrist','head_translation_matched','active_wrist']:
 for h in ['left','right','pooled']:
    b=res[t][h];print(t,h,b['n'],b['n_episodes'],b.get('translation_cm'),b.get('rotation_deg'))
