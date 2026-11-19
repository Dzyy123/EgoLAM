"""Native LARY attentive probe with only task-specific outputs changed."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class AbilityProbe(nn.Module):
    def __init__(self,lary_root,input_dim,task,steps=8,nverbs=0,nnouns=0,context_dim=None,
                 model_dim=768,heads=16,depth=4,condition='full'):
        super().__init__();sys.path.insert(0,str(Path(lary_root).resolve()))
        from classification.src.models.attentive_pooler import FeatureEvaluator,AttentivePooler
        self.task=task;self.steps=steps;self.nverbs=nverbs;self.nnouns=nnouns;self.condition=condition
        outputs={'effect':2,'pnr':2*steps,'composition':nverbs+nnouns,'spatial':4}[task]
        if task=='composition' and min(nverbs,nnouns)<2:raise ValueError('At least two components per vocabulary')
        if task!='spatial':
            self.native=FeatureEvaluator(input_dim=input_dim,model_dim=model_dim,num_classes=outputs,
                num_heads=heads,depth=depth)
        else:
            if context_dim is None:raise ValueError('Spatial task needs shared current-image context')
            self.z_projector=nn.Sequential(nn.Linear(input_dim,model_dim),nn.LayerNorm(model_dim),nn.GELU())
            self.context_projector=nn.Sequential(nn.Linear(context_dim,model_dim),nn.LayerNorm(model_dim),nn.GELU())
            self.context_position=nn.Linear(2,model_dim)
            self.time_embedding=nn.Parameter(torch.zeros(1,steps,1,model_dim))
            nn.init.normal_(self.time_embedding,std=.02)
            self.modality=nn.Parameter(torch.zeros(2,model_dim))
            self.pooler=AttentivePooler(embed_dim=model_dim,num_heads=heads,depth=depth)
            self.head=nn.Linear(model_dim,4)

    def forward(self,z,context=None,context_xy=None):
        if z.ndim!=4 or z.shape[1]!=self.steps:raise ValueError('Expected fixed [B,T,K,D] latent sequence')
        if self.task!='spatial':return self.native(z)
        if context is None or context_xy is None:raise ValueError('Spatial features/positions missing')
        c=self.context_projector(context)+self.context_position(context_xy)+self.modality[0]
        if self.condition=='context_only':tokens=c
        else:
            zz=self.z_projector(z)+self.time_embedding+self.modality[1]
            tokens=torch.cat([c,zz.flatten(1,2)],dim=1)
        raw=self.head(self.pooler(tokens).squeeze(1)).sigmoid()
        center,size=raw[:,:2],raw[:,2:]
        return torch.cat([(center-size/2).clamp(0,1),(center+size/2).clamp(0,1)],dim=-1)

    def loss(self,pred,target):
        if self.task=='effect':return F.cross_entropy(pred.float(),target['label'].long())
        if self.task=='composition':
            return .5*(F.cross_entropy(pred[:,:self.nverbs].float(),target['verb'].long())+
                       F.cross_entropy(pred[:,self.nverbs:].float(),target['noun'].long()))
        if self.task=='pnr':
            b=target['bin'].long();off=pred[:,self.steps:].float().tanh()[torch.arange(len(b),device=b.device),b]
            return F.cross_entropy(pred[:,:self.steps].float(),b)+.1*F.smooth_l1_loss(off,target['offset'].float())
        y=target['box'].float();p=pred.float()
        inter=(torch.minimum(p[:,2:],y[:,2:])-torch.maximum(p[:,:2],y[:,:2])).clamp_min(0).prod(1)
        union=(p[:,2:]-p[:,:2]).prod(1)+(y[:,2:]-y[:,:2]).prod(1)-inter
        enclosing=(torch.maximum(p[:,2:],y[:,2:])-torch.minimum(p[:,:2],y[:,:2])).clamp_min(1e-8).prod(1)
        giou=inter/union.clamp_min(1e-8)-(enclosing-union)/enclosing
        return F.l1_loss(p,y)+(1-giou).mean()
