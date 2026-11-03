"""Tensor-native spatial compression. Representatives are not object labels."""
import math
import torch
import torch.nn.functional as F

def dilate(mask):
    return F.max_pool2d(mask.float()[None,None],3,1,1)[0,0].bool()

def _window_indices(h,w,device):
    # Geometry only: no data-dependent device synchronization/component loop.
    rows=[];sizes=[]
    for y in range(0,h,4):
        for x in range(0,w,4):
            ids=[a*w+b for a in range(y,min(y+4,h)) for b in range(x,min(x+4,w))]
            sizes.append(len(ids));rows.append(ids+[-1]*(16-len(ids)))
    return torch.tensor(rows,device=device),sizes

def _counts(sizes,target,device):
    # Fixed spatial coverage floor, then deterministic balanced allocation.
    result=[1]*len(sizes);remaining=max(0,target-len(sizes))
    while remaining:
        changed=False
        for i,n in enumerate(sizes):
            if result[i]<n and remaining:result[i]+=1;remaining-=1;changed=True
        if not changed:break
    return torch.tensor(result,device=device)[:,None]

@torch.inference_mode()
def build_plan(features,*,image_tokens=0,retention=.30,mode='prune_merge',key_period=8,signal='novelty'):
    if features.ndim!=4 or min(features.shape)<=0 or mode not in ('select','merge','prune_merge') or signal not in ('novelty','norm','random') or not 0<retention<=1 or key_period<1 or image_tokens<0:
        raise ValueError('invalid compression contract')
    if not bool(torch.isfinite(features).all()):raise ValueError('nonfinite visual features')
    t,h,w,d=features.shape;n=t*h*w;device=features.device
    keys=sorted(set([0,t-1]+list(range(0,t,key_period))))
    win,sizes=_window_indices(h,w,device);valid=win>=0;safe=win.clamp_min(0)
    desired=math.floor((n+image_tokens)*retention)-image_tokens
    nonkeys=t-len(keys);floor=len(keys)*h*w+nonkeys*len(sizes)
    actual_target=min(n,max(floor,desired))
    remaining=actual_target-len(keys)*h*w
    unit,remainder=divmod(remaining,max(1,nonkeys));ordinal=0
    ref=F.normalize(features[0].float(),dim=-1).flatten(0,1).clone()
    all_selected=[];all_assigned=[];all_novelty=[];all_protected=[];all_scores=[];all_norms=[]
    geometry=torch.arange(h*w,device=device)
    for ti in range(t):
        cur=F.normalize(features[ti].float(),dim=-1).flatten(0,1)
        novelty=(1-(cur*ref).sum(-1)).clamp(0,2)
        norms=features[ti].float().flatten(0,1).norm(dim=-1)
        if signal=='novelty':score=novelty
        elif signal=='norm':score=norms/norms.max().clamp_min(1e-12)
        else:
            # Stateless integer hash: fixed seed, reproducible CPU/CUDA, no truth.
            rng=(geometry+ti*h*w+20261023)*1103515245+12345
            rng=(rng^(rng>>16))*1103515245+12345
            score=(rng&0x7fffffff).float()/2147483648.
        if ti in keys or retention==1:
            selected=torch.ones(h*w,device=device,dtype=torch.bool)
            protected=selected.clone();assignment=geometry+ti*h*w
        else:
            protected=dilate((novelty>=.10).reshape(h,w)).flatten()
            priority=score+protected.float()*2
            ranks=torch.argsort(priority[safe].masked_fill(~valid,-float('inf')),dim=1,descending=True,stable=True)
            counts=_counts(sizes,unit+(ordinal<remainder),device);ordinal+=1
            chosen=torch.zeros_like(valid).scatter_(1,ranks,torch.arange(16,device=device)[None,:]<counts)&valid
            selected=torch.zeros(h*w,device=device,dtype=torch.bool)
            selected[win[chosen]]=True
            vectors=cur[safe]
            similarity=torch.bmm(vectors,vectors.transpose(1,2))
            nearest=similarity.masked_fill(~chosen[:,None,:],-float('inf')).argmax(-1)
            target=win.gather(1,nearest)
            assignment=torch.empty(h*w,device=device,dtype=torch.long)
            assignment[win[valid]]=target[valid]+ti*h*w
            assignment[selected]=geometry[selected]+ti*h*w
        ref=torch.where(selected[:,None],cur,ref)
        if mode=='select':assignment=assignment.masked_fill(~selected,-1)
        elif mode=='prune_merge':assignment=assignment.masked_fill(~(selected|protected),-1)
        all_selected.append(selected);all_assigned.append(assignment)
        all_novelty.append(novelty);all_protected.append(protected)
        all_scores.append(score);all_norms.append(norms)
    mask=torch.cat(all_selected);centers=torch.where(mask)[0]
    assigned=torch.cat(all_assigned);present=assigned>=0
    reverse=torch.full((n,),-1,device=device,dtype=torch.long);reverse[centers]=torch.arange(centers.numel(),device=device)
    assignment=torch.where(present,reverse[assigned.clamp_min(0)],-1)
    novelty=torch.stack(all_novelty).reshape(t,h,w)
    return dict(centers=centers,assignment=assignment,weights=novelty.flatten()+.01,
        novelty=novelty,selection_signal=torch.stack(all_scores).reshape(t,h,w),feature_norm=torch.stack(all_norms).reshape(t,h,w),signal=signal,
        protected_mask=torch.stack(all_protected).reshape(t,h,w),selected_mask=mask,
        grid_thw=[t,h,w],original_tokens=n,mode=mode,key_groups=keys,requested_video_budget=desired,
        budget_floor=floor,budget_exceeded=centers.numel()>desired,identity=centers.numel()==n,
        target_visual_retention=retention)

@torch.inference_mode()
def apply_plan(main,deepstack,plan):
    if len(deepstack)!=3 or any(x.shape!=main.shape for x in deepstack) or len(main)!=plan['original_tokens']:
        raise ValueError('all four feature streams must align')
    centers=plan['centers']
    if plan['identity']:return main,deepstack
    if plan['mode']=='select':return main[centers],[x[centers] for x in deepstack]
    assignment=plan['assignment'];active=assignment>=0;slots=assignment[active];weights=plan['weights'][active]
    denom=torch.zeros(len(centers),device=main.device,dtype=torch.float32).index_add_(0,slots,weights)
    def merge(x):
        result=torch.zeros((len(centers),x.shape[-1]),device=x.device,dtype=torch.float32)
        result.index_add_(0,slots,x[active].float()*weights[:,None])
        # Preserve singleton representatives exactly, avoiding needless rounding.
        result=(result/denom[:,None]).to(x.dtype)
        counts=torch.zeros(len(centers),device=x.device,dtype=torch.long).index_add_(0,slots,torch.ones_like(slots))
        return torch.where((counts==1)[:,None],x[centers],result)
    return merge(main),[merge(x) for x in deepstack]
