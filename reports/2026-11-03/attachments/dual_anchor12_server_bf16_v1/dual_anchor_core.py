"""Frozen dual-anchor signals and bounded main-only residual, no reference labels."""
import math
import torch
import torch.nn.functional as F
from high_compression_core import geometry_maps,build_high_plan
from purity_core import spatial_topk

def dual_signals(features):
    x=F.normalize(features.float(),dim=-1)
    previous=torch.cat([x[:1],x[:-1]],0)
    a=(1-(x*previous).sum(-1)).clamp(0,2)
    b=(1-(x*x[:1]).sum(-1)).clamp(0,2)
    n=.7*a+.3*b
    n[0]=0
    return n

def strong_cross(novelty):
    if novelty.ndim!=2:raise ValueError('one temporal spatial grid required')
    seed=novelty>=.18
    result=seed.clone()
    result[1:]|=seed[:-1];result[:-1]|=seed[1:]
    result[:,1:]|=seed[:,:-1];result[:,:-1]|=seed[:,1:]
    return result

def residual_sources(features,centers,protected,novelty):
    t,h,w,d=features.shape;hw=h*w;n=t*hw;device=features.device
    assignment=torch.full((n,),-1,device=device,dtype=torch.long)
    coefficient=torch.zeros(n,device=device,dtype=torch.float32)
    assignment[centers]=torch.arange(len(centers),device=device);coefficient[centers]=1
    ids=torch.arange(hw,device=device)
    normalized=F.normalize(features.float(),dim=-1).reshape(t,hw,d)
    for ti in range(1,t):
        slots=torch.where(centers//hw==ti)[0]
        if not len(slots):raise ValueError('each time group requires a representative')
        local=centers[slots]%hw
        similarity=normalized[ti]@normalized[ti,local].T
        distance=torch.maximum((ids[:,None]//w-local[None,:]//w).abs(),(ids[:,None]%w-local[None,:]%w).abs())
        eligible=(distance<=2)&(similarity>=.85)&protected[ti].flatten()[:,None]
        eligible[local]=False
        ranking=similarity.masked_fill(~eligible,-float('inf'))
        winning=ranking.argmax(-1);active=eligible.any(-1)
        source=ids[active]+ti*hw;target=slots[winning[active]]
        weight=novelty[ti].flatten()[active]+.01
        denominator=torch.zeros(len(centers),device=device).index_add_(0,target,weight)
        assignment[source]=target
        coefficient[source]=.25*weight/denominator[target]
        coefficient[centers]=torch.where(denominator>0,torch.full_like(denominator,.75),coefficient[centers])
    return assignment,coefficient

@torch.inference_mode()
def build_dual_plan(features,*,image_tokens=0,retention=.1,residual=False):
    if features.ndim!=4 or min(features.shape)<=0 or not 0<retention<=1 or image_tokens<0 or not bool(torch.isfinite(features).all()):
        raise ValueError('invalid dual-anchor input contract')
    t,h,w,d=features.shape;hw=h*w;n=t*hw;device=features.device
    scene,_,_,sizes=geometry_maps(h,w,device)
    desired=math.floor((n+image_tokens)*retention)-image_tokens
    floor=len(scene)+(t-1)*len(sizes);target=min(n,max(floor,desired))
    novelty=dual_signals(features)
    if retention==1 or target==n:
        p=build_high_plan(features,image_tokens=image_tokens,retention=1.,mode='prune_merge')
        p.update(policy='dual_anchor_v1',variant='dual_residual90' if residual else 'dual90',
          main_source_coefficients=torch.ones(n,device=device),novelty=novelty,selection_signal=novelty)
        return p
    if t<2:raise ValueError('compressed diagnostic videos require at least two temporal groups')
    unit,remainder=divmod(target-len(scene),t-1);masks=[];protections=[]
    for ti in range(t):
        if ti==0:
            selected=torch.zeros(hw,device=device,dtype=torch.bool);selected[scene]=True
            protected=torch.ones((h,w),device=device,dtype=torch.bool)
        else:
            protected=strong_cross(novelty[ti])
            selected=spatial_topk(novelty[ti].flatten()+2*protected.flatten().float(),h,w,unit+(ti-1<remainder))
        masks.append(selected);protections.append(protected)
    selected=torch.cat(masks);centers=torch.where(selected)[0];protected=torch.stack(protections)
    if len(centers)!=target:raise ValueError('registered exact budget changed')
    if residual:assignment,coefficient=residual_sources(features,centers,protected,novelty)
    else:
        assignment=torch.full((n,),-1,device=device,dtype=torch.long);assignment[centers]=torch.arange(len(centers),device=device)
        coefficient=selected.float()
    return dict(centers=centers,assignment=assignment,weights=coefficient,main_source_coefficients=coefficient,
      novelty=novelty,selection_signal=novelty,protected_mask=protected,selected_mask=selected,grid_thw=[t,h,w],
      original_tokens=n,mode='residual' if residual else 'select',identity=False,scene_groups=[0],scene_tokens=len(scene),
      key_groups=[],policy='dual_anchor_v1',variant='dual_residual90' if residual else 'dual90',
      requested_video_budget=desired,budget_floor=floor,budget_exceeded=target>desired,target_visual_retention=retention,
      previous_weight=.7,first_weight=.3,strong_threshold=.18,dilation='cross4',residual_alpha=.25 if residual else 0.,
      residual_cosine_min=.85,residual_chebyshev_radius=2,deepstack_original_centers=True)

@torch.inference_mode()
def apply_dual_plan(main,deepstack,plan):
    if len(deepstack)!=3 or any(x.shape!=main.shape for x in deepstack) or len(main)!=plan['original_tokens']:
        raise ValueError('all four streams must align')
    if plan['identity']:return main,deepstack
    centers=plan['centers'];original=main[centers]
    if plan['mode']=='select':return original,[x[centers] for x in deepstack]
    active=plan['assignment']>=0;slots=plan['assignment'][active];weights=plan['main_source_coefficients'][active]
    merged=torch.zeros((len(centers),main.shape[-1]),device=main.device,dtype=torch.float32)
    merged.index_add_(0,slots,main[active].float()*weights[:,None])
    singleton=plan['main_source_coefficients'][centers]==1
    merged=torch.where(singleton[:,None],original,merged.to(main.dtype))
    return merged,[x[centers] for x in deepstack]
