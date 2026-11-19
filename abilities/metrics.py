import numpy as np


def classification(pred,target,classes,include_confusion=True):
    p,y=np.asarray(pred,int),np.asarray(target,int)
    if p.shape!=y.shape or p.ndim!=1 or ((p<0)|(p>=classes)|(y<0)|(y>=classes)).any():
        raise ValueError('Invalid classification labels/shapes')
    gt=np.bincount(y,minlength=classes);det=np.bincount(p,minlength=classes)
    tp=np.bincount(y[p==y],minlength=classes);den=gt+det
    f1=np.divide(2*tp,den,out=np.zeros(classes,float),where=den>0)
    recall=np.divide(tp,gt,out=np.zeros(classes,float),where=gt>0)
    present=gt>0
    out={'n':len(y),'accuracy':float((p==y).mean()) if len(y) else None,
        'macro_f1_present':float(f1[present].mean()) if present.any() else None,
        'balanced_accuracy':float(recall[present].mean()) if present.any() else None,
        'class_support':gt.tolist()}
    if include_confusion and classes<=512:
        cm=np.zeros((classes,classes),np.int64);np.add.at(cm,(y,p),1);out['confusion']=cm.tolist()
    return out


def temporal(pred_sec,true_sec,duration_sec):
    p,y,d=map(lambda a:np.asarray(a,float),(pred_sec,true_sec,duration_sec))
    if p.shape!=y.shape or p.shape!=d.shape or (d<=0).any() or not np.isfinite([p,y,d]).all():
        raise ValueError('Invalid PNR times/durations')
    e=abs(p-y)
    return {'n':len(e),'mae_seconds':float(e.mean()) if len(e) else None,
        'median_seconds':float(np.median(e)) if len(e) else None,
        'mean_duration_normalized_error':float((e/d).mean()) if len(e) else None,
        'hit_025s':float((e<=.25).mean()) if len(e) else None,
        'hit_05s':float((e<=.5).mean()) if len(e) else None}


def box_iou(pred,true):
    p,y=np.asarray(pred,float),np.asarray(true,float)
    if p.shape!=y.shape or p.ndim!=2 or p.shape[1]!=4 or not np.isfinite([p,y]).all():raise ValueError('Invalid boxes')
    if ((p<0)|(p>1)|(y<0)|(y>1)).any() or (p[:,2:]<p[:,:2]).any() or (y[:,2:]<=y[:,:2]).any():raise ValueError('Boxes must be normalized xyxy')
    inter=np.maximum(np.minimum(p[:,2:],y[:,2:])-np.maximum(p[:,:2],y[:,:2]),0).prod(1)
    union=(p[:,2:]-p[:,:2]).prod(1)+(y[:,2:]-y[:,:2]).prod(1)-inter
    return inter/np.maximum(union,1e-12)


def localization(pred,true):
    iou=box_iou(pred,true)
    return {'n':len(iou),'mean_iou':float(iou.mean()) if len(iou) else None,
        'recall_iou05':float((iou>=.5).mean()) if len(iou) else None,
        'recall_iou075':float((iou>=.75).mean()) if len(iou) else None,
        'scope':'single affected-object localization; NOT full detection AP'}


def composition(pv,pn,yv,yn,unseen,nverbs,nnouns):
    pv,pn,yv,yn=map(lambda x:np.asarray(x,int),(pv,pn,yv,yn));u=np.asarray(unseen,bool)
    out={'verb':classification(pv,yv,nverbs),'noun':classification(pn,yn,nnouns)}
    for name,mask in [('all',np.ones(len(u),bool)),('seen',~u),('unseen',u)]:
        out[name]=classification((pv*nnouns+pn)[mask],(yv*nnouns+yn)[mask],nverbs*nnouns,False)
    return out


def pnr_targets(edges,truth):
    edges=np.asarray(edges,float);truth=np.asarray(truth,float)
    if edges.ndim!=2 or truth.shape!=(len(edges),) or (np.diff(edges,axis=1)<=0).any():raise ValueError('Strictly increasing frame-time edges required')
    if not np.isfinite(edges).all() or not np.isfinite(truth).all() or (truth<edges[:,0]).any() or (truth>edges[:,-1]).any():raise ValueError('PNR outside clip')
    bins=np.array([min(len(e)-2,np.searchsorted(e,t,side='right')-1) for e,t in zip(edges,truth)])
    i=np.arange(len(bins));mid=(edges[i,bins]+edges[i,bins+1])/2;half=(edges[i,bins+1]-edges[i,bins])/2
    return bins,(truth-mid)/half
