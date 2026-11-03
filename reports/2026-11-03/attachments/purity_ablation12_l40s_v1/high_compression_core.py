"""High compression with a 64-bin scene and 8x8 local later windows."""
import math
import torch
import torch.nn.functional as F
from prune_merge_core import dilate,build_plan,_counts

def geometry_maps(h,w,device):
    scene_target=[-1]*(h*w);scene_centers=[]
    for by in range(min(8,h)):
        y0=by*h//min(8,h);y1=(by+1)*h//min(8,h)
        for bx in range(min(8,w)):
            x0=bx*w//min(8,w);x1=(bx+1)*w//min(8,w)
            center=((y0+y1-1)//2)*w+(x0+x1-1)//2;scene_centers.append(center)
            for y in range(y0,y1):
                for x in range(x0,x1):scene_target[y*w+x]=center
    windows=[];sizes=[]
    for y in range(0,h,8):
        for x in range(0,w,8):
            ids=[a*w+b for a in range(y,min(y+8,h)) for b in range(x,min(x+8,w))]
            sizes.append(len(ids));windows.append(ids+[-1]*(64-len(ids)))
    return torch.tensor(scene_centers,device=device),torch.tensor(scene_target,device=device),torch.tensor(windows,device=device),sizes

@torch.inference_mode()
def build_high_plan(features,*,image_tokens=0,retention=.08,mode='prune_merge'):
    if features.ndim!=4 or min(features.shape)<=0 or mode not in ('merge','prune_merge') or not 0<retention<=1 or image_tokens<0:raise ValueError('invalid high-compression contract')
    if not bool(torch.isfinite(features).all()):raise ValueError('nonfinite features')
    if retention==1:
        p=build_plan(features,image_tokens=image_tokens,retention=1.,mode=mode)
        p.update(scene_groups=[0],policy='high_scene64_window8_v1');return p
    t,h,w,d=features.shape;hw=h*w;n=t*hw;device=features.device
    scene,scene_target,win,sizes=geometry_maps(h,w,device);valid=win>=0;safe=win.clamp_min(0)
    desired=math.floor((n+image_tokens)*retention)-image_tokens
    floor=len(scene)+(t-1)*len(sizes);target=min(n,max(floor,desired))
    q,rem=divmod(target-len(scene),max(1,t-1))
    ref=F.normalize(features[0].float(),dim=-1).flatten(0,1).clone();geometry=torch.arange(hw,device=device)
    selected_all=[];assigned_all=[];novelties=[];protected_all=[];norms_all=[]
    for ti in range(t):
        cur=F.normalize(features[ti].float(),dim=-1).flatten(0,1)
        novelty=(1-(cur*ref).sum(-1)).clamp(0,2)
        if ti==0:
            selected=torch.zeros(hw,device=device,dtype=torch.bool);selected[scene]=True
            protected=torch.ones_like(selected);assignment=scene_target.clone()
        else:
            protected=dilate((novelty>=.1).reshape(h,w)).flatten()
            score=novelty+2*protected.float()
            ranks=torch.argsort(score[safe].masked_fill(~valid,-float('inf')),dim=1,descending=True,stable=True)
            count=_counts(sizes,q+(ti-1<rem),device)
            chosen=torch.zeros_like(valid).scatter_(1,ranks,torch.arange(64,device=device)[None,:]<count)&valid
            selected=torch.zeros(hw,device=device,dtype=torch.bool);selected[win[chosen]]=True
            vectors=cur[safe];similarity=torch.bmm(vectors,vectors.transpose(1,2))
            nearest=similarity.masked_fill(~chosen[:,None,:],-float('inf')).argmax(-1)
            targets=win.gather(1,nearest);assignment=torch.empty(hw,device=device,dtype=torch.long)
            assignment[win[valid]]=targets[valid]+ti*hw
            assignment[selected]=geometry[selected]+ti*hw
        ref=torch.where(selected[:,None],cur,ref)
        if mode=='prune_merge':assignment=assignment.masked_fill(~(selected|protected),-1)
        selected_all.append(selected);assigned_all.append(assignment);novelties.append(novelty);protected_all.append(protected)
        norms_all.append(features[ti].float().flatten(0,1).norm(dim=-1))
    mask=torch.cat(selected_all);centers=torch.where(mask)[0];targets=torch.cat(assigned_all);active=targets>=0
    reverse=torch.full((n,),-1,device=device,dtype=torch.long);reverse[centers]=torch.arange(len(centers),device=device)
    assignment=torch.where(active,reverse[targets.clamp_min(0)],-1);novelty=torch.stack(novelties).reshape(t,h,w)
    return dict(centers=centers,assignment=assignment,weights=novelty.flatten()+.01,novelty=novelty,
        selection_signal=novelty,feature_norm=torch.stack(norms_all).reshape(t,h,w),signal='novelty',
        protected_mask=torch.stack(protected_all).reshape(t,h,w),selected_mask=mask,grid_thw=[t,h,w],original_tokens=n,
        mode=mode,key_groups=[],scene_groups=[0],scene_tokens=len(scene),policy='high_scene64_window8_v1',
        requested_video_budget=desired,budget_floor=floor,budget_exceeded=len(centers)>desired,identity=len(centers)==n,target_visual_retention=retention)
