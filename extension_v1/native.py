"""Reuse pinned native model and scheduler code; small task-specific output/loss."""
import ast,hashlib,math,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_COMMIT='e7feaf1b72921ee2c34e489adb0f45faf356ecee'
GRID=[(.005,.01),(.003,.1),(.001,.4),(.0003,.8),(.0001,.01)]

def schedules(path):
 tree=ast.parse(Path(path).read_text());names={'WarmupCosineLRSchedule','CosineWDSchedule'}
 classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in names]
 if {x.name for x in classes}!=names:raise ValueError('Pinned scheduler definitions missing')
 namespace={'math':math};exec(compile(ast.Module(body=classes,type_ignores=[]),str(path),'exec'),namespace)
 return namespace['WarmupCosineLRSchedule'],namespace['CosineWDSchedule']

class Probe(nn.Module):
 def __init__(self,lary_root,input_dim,task,nverbs=0,nnouns=0):
  super().__init__();sys.path.insert(0,str(lary_root))
  from classification.src.models.attentive_pooler import FeatureEvaluator
  self.task=task;self.nverbs=nverbs;self.nnouns=nnouns
  self.native=FeatureEvaluator(input_dim=input_dim,model_dim=768,num_classes=2 if task=='effect' else nverbs+nnouns,num_heads=16,depth=4,use_activation_checkpointing=True)
 def forward(self,x):return self.native(x)
 def loss(self,p,y):
  if self.task=='effect':return F.cross_entropy(p.float(),y['label'],label_smoothing=.1)
  return .5*(F.cross_entropy(p[:,:self.nverbs].float(),y['verb'],label_smoothing=.1)+F.cross_entropy(p[:,self.nverbs:].float(),y['noun'],label_smoothing=.1))

def optimizer(net,lr,wd,steps,scheduler_source):
 lr_schedule,wd_schedule=schedules(scheduler_source)
 group={'params':net.parameters(),'mc_warmup_steps':0,'mc_start_lr':lr,'mc_ref_lr':lr,'mc_final_lr':0.,'mc_ref_wd':wd,'mc_final_wd':wd}
 opt=torch.optim.AdamW([group])
 return opt,lr_schedule(opt,steps),wd_schedule(opt,steps)
