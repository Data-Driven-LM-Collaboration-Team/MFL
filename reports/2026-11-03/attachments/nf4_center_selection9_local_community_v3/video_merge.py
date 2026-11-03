"""Exact video feature merge functions extracted from the frozen local v5 method."""
import math
import torch

def merge_plan(features,grid_thw,ratio):
    if not 0<ratio<=1:raise ValueError('ratio outside (0,1]')
    f=torch.nn.functional.normalize(features.detach().to(device='cpu',dtype=torch.float32),dim=-1)
    if not bool(torch.isfinite(f).all()):raise ValueError('Nonfinite feature')
    grids=grid_thw.detach().cpu().tolist();expected=sum(t*(h//2)*(w//2) for t,h,w in grids)
    if any(h%2 or w%2 for t,h,w in grids) or len(f)!=expected:raise ValueError('Grid count mismatch')
    centers=[];groups=[];offset=0
    for t,h,w in grids:
        h//=2;w//=2;hw=h*w;previous=None
        for frame in range(t):
            x=f[offset:offset+hw];budget=hw if frame in (0,t-1) else max(1,math.ceil(hw*ratio))
            if budget==hw:
                local=list(range(hw));localgroups=[[i] for i in local]
            else:
                novelty=1-(x*previous).sum(-1)
                # A spatial floor survives even in static areas; no wholesale
                # deletion of stationary people, fire, parked objects or context.
                floor=list(range(0,hw,16))[:budget]
                candidates=sorted(set(floor+[i for i in range(hw) if float(novelty[i])>.1]))
                for i in sorted(range(hw),key=lambda i:(-float(novelty[i]),i)):
                    if len(candidates)>=budget:break
                    if i not in candidates:candidates.append(i)
                candidates=sorted(candidates);pool=x[candidates];m=len(candidates)
                coords=torch.tensor([[i//w,i%w] for i in candidates],dtype=torch.float32)
                cosine=(1-pool@pool.T).clamp_min(0)
                distance=cosine+.05*torch.cdist(coords,coords)/max(h,w)
                ds=distance.clone();ds.fill_diagonal_(float('inf'))
                k=min(5,m-1)
                density=torch.exp(-ds.topk(k,largest=False).values.square().mean(-1)) if k else torch.ones(m)
                order=sorted(range(m),key=lambda i:(-float(density[i]),candidates[i]))
                delta=torch.zeros(m)
                for rank,i in enumerate(order):
                    delta[i]=distance[i,order[:rank]].min() if rank else distance[i].max()
                ranking=sorted(range(m),key=lambda i:(-float(density[i]*delta[i]),candidates[i]))
                local=list(floor)
                for i in ranking:
                    if len(local)==budget:break
                    if candidates[i] not in local:local.append(candidates[i])
                local=sorted(local);buckets={i:[i] for i in local}
                # Uniform feature merging is spatially local and same-frame.
                # Far/disagreeing features are dropped and recorded, not mixed.
                for i in candidates:
                    if i in buckets:continue
                    choices=[j for j in local if abs(i//w-j//w)<=2 and abs(i%w-j%w)<=2]
                    if not choices:continue
                    j=max(choices,key=lambda j:(float(x[i]@x[j]),-j))
                    if float(x[i]@x[j])>=.7:buckets[j].append(i)
                localgroups=[sorted(buckets[i]) for i in local]
            centers.extend(offset+i for i in local)
            groups.extend([[offset+i for i in g] for g in localgroups])
            previous=x;offset+=hw
    members=set(i for g in groups for i in g)
    return dict(centers=centers,groups=groups,pruned_indices=[i for i in range(len(f)) if i not in members],
        requested_interior_ratio=ratio,actual_retained_ratio=len(centers)/len(f),
        first_last_time_groups_preserved=True,spatial_floor_stride=16,
        clusters_are_object_ids=False)

def merge_features(features,plan):
    return torch.stack([features[g[0]] if len(g)==1 else features[g].float().mean(0).to(features.dtype) for g in plan['groups']])
