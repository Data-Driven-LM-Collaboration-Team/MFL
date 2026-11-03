"""Matched-budget pure original-vector selection; no object labels or averaging."""
import torch
import torch.nn.functional as F
from high_compression_core import build_high_plan,geometry_maps
from prune_merge_core import dilate

def spatial_topk(score,h,w,count):
    """Stable greedy selection: at most two centers in each center's radius-2 box."""
    hw=h*w;ids=torch.arange(hw,device=score.device)
    selected=torch.zeros(hw,dtype=torch.bool,device=score.device)
    neighbours=torch.zeros(hw,dtype=torch.int32,device=score.device)
    blocked=torch.zeros_like(selected)
    for _ in range(count):
        valid=(~selected)&(~blocked)&(neighbours<2)
        ranked=score.masked_fill(~valid,-float('inf'))
        idx=ranked.argmax()
        if not bool(torch.isfinite(ranked[idx])):
            raise ValueError('registered spatial exclusion cannot satisfy budget')
        selected[idx]=True
        neighbours+=(((ids//w-idx//w).abs()<=2)&((ids%w-idx%w).abs()<=2)).int()
        saturated=selected&(neighbours>=2)
        blocked=F.max_pool2d(saturated.float().reshape(1,1,h,w),5,1,2).flatten().bool()
    return selected

@torch.inference_mode()
def build_purity_plan(features,*,image_tokens=0,retention=.1,variant='select90'):
    if variant not in ('select90','global90','anchor90'):
        raise ValueError('unregistered purity variant')
    # A must use the unmodified parent algorithm, including its selected-cache dynamics.
    p=build_high_plan(features,image_tokens=image_tokens,retention=retention,mode='prune_merge')
    t,h,w,d=features.shape;hw=h*w;n=t*hw;device=features.device
    if variant!='select90' and not p['identity']:
        target=len(p['centers']);scene,_,_,_=geometry_maps(h,w,device)
        first=len(scene)
        if variant=='anchor90':
            first=min(max(first,(hw+1)//2),target-(t-1))
            scene=torch.div(torch.arange(first,device=device)*hw,first,rounding_mode='floor')
        q,rem=divmod(target-first,max(1,t-1))
        ref=F.normalize(features[0].float(),dim=-1).reshape(hw,d).clone()
        masks=[];novelties=[];protections=[]
        for ti in range(t):
            cur=F.normalize(features[ti].float(),dim=-1).reshape(hw,d)
            novelty=(1-(cur*ref).sum(-1)).clamp(0,2)
            if ti==0:
                selected=torch.zeros(hw,device=device,dtype=torch.bool);selected[scene]=True
                protected=torch.ones_like(selected)
            else:
                protected=dilate((novelty>=.1).reshape(h,w)).flatten()
                selected=spatial_topk(novelty+2*protected.float(),h,w,q+(ti-1<rem))
            ref=torch.where(selected[:,None],cur,ref)
            masks.append(selected);novelties.append(novelty);protections.append(protected)
        mask=torch.cat(masks);centers=torch.where(mask)[0]
        if len(centers)!=target:raise ValueError('matched budget changed')
        p.update(centers=centers,selected_mask=mask,novelty=torch.stack(novelties).reshape(t,h,w),
            protected_mask=torch.stack(protections).reshape(t,h,w),scene_tokens=first)
    centers=p['centers'];assignment=torch.full((n,),-1,device=device,dtype=torch.long)
    assignment[centers]=torch.arange(len(centers),device=device)
    p.update(assignment=assignment,mode='select',weights=torch.ones(n,device=device),
        selection_signal=p['novelty'],policy='purity_ablation_v1',variant=variant,
        exact_original_main_and_deepstack=True,selected_cache_rule_unchanged=True,
        neighbour_radius=2 if variant!='select90' else None,neighbour_limit_including_self=2 if variant!='select90' else None)
    return p
