"""User-specified exterior operators, not yet connected to a research runner.

Components are feature components, not verified physical objects. Retention is
measured, never obtained by refilling pruned background to a requested ratio.
"""
import math
import torch
import torch.nn.functional as F


def _dpc_groups(features, indices, count=2):
    n=len(indices)
    if n==1: return [indices[0]],[indices]
    local=features[indices]
    distance=torch.cdist(local,local,p=2)
    neighbor=distance.clone(); neighbor.fill_diagonal_(float('inf'))
    k=min(5,n-1)
    rho=torch.exp(-neighbor.topk(k,largest=False).values.square().mean(dim=1))
    densities=rho.detach().cpu().tolist()
    order=sorted(range(n),key=lambda i:(-densities[i],indices[i]))
    delta=torch.empty_like(rho)
    for rank,i in enumerate(order):
        delta[i]=distance[i].max() if rank==0 else distance[i,order[:rank]].min()
    score=(rho*delta).detach().cpu().tolist()
    centers=sorted(sorted(range(n),key=lambda i:(-score[i],indices[i]))[:count],key=lambda i:indices[i])
    assignment=distance[:,centers].argmin(dim=1).detach().cpu().tolist()
    # Equal-feature ties must not silently delete a selected center.
    for slot,center in enumerate(centers): assignment[center]=slot
    groups=[[indices[i] for i in range(n) if assignment[i]==slot] for slot in range(len(centers))]
    return [indices[i] for i in centers],groups


def build_scene_plan(main, background_threshold=.90, object_threshold=.80,*,stage_mode='combined',scene_mode='literal2x2',target_tokens=None):
    if (not isinstance(main,torch.Tensor) or main.ndim!=4 or min(main.shape)<=0
            or not main.is_floating_point() or not torch.isfinite(main).all().item()):
        raise ValueError('expected finite floating [T,H,W,D] features')
    for value in (background_threshold,object_threshold):
        if type(value) not in (int,float) or not math.isfinite(value) or not -1<=value<=1:
            raise ValueError('invalid cosine threshold')
    tcount,height,width,dim=main.shape
    if stage_mode not in ('combined','pruner_only','merger_only') or scene_mode not in ('literal2x2','fixed64'):
        raise ValueError('unregistered compression ablation')
    if target_tokens is not None and (type(target_tokens) is not int or target_tokens<1):raise ValueError('positive explicit token budget required')
    hw=height*width
    normalized=F.normalize(main.detach().float(),p=2,dim=-1,eps=1e-12)
    flat=normalized.reshape(-1,dim)
    groups=[]; centers=[]; time_groups=[]; kinds=[]; component_ids=[]
    pruned=[]; components=[]
    # Arithmetic 2x2 spatial pooling, including partial boundary blocks.
    if scene_mode=='fixed64':
        if min(height,width)<8:raise ValueError('fixed64 scene grid requires at least 8x8 native tokens')
        rows=[(i*height//8,(i+1)*height//8) for i in range(8)]
        cols=[(i*width//8,(i+1)*width//8) for i in range(8)]
    else:
        rows=[(r,min(r+2,height)) for r in range(0,height,2)]
        cols=[(c,min(c+2,width)) for c in range(0,width,2)]
    for r,yend in rows:
        for c,xend in cols:
            g=[y*width+x for y in range(r,yend) for x in range(c,xend)]
            cy=sum(i//width for i in g)/len(g); cx=sum(i%width for i in g)/len(g)
            center=min(g,key=lambda i:((i//width-cy)**2+(i%width-cx)**2,i))
            groups.append(g); centers.append(center); time_groups.append(0)
            kinds.append('scene'); component_ids.append(None)
    scene_count=len(groups)
    for t in range(1,tcount):
        similarity=(normalized[t]*normalized[t-1]).sum(-1)
        keep=torch.ones_like(similarity,dtype=torch.bool) if stage_mode=='merger_only' else similarity<=background_threshold
        keep_cpu=keep.detach().cpu()
        selected=[i for i in range(hw) if bool(keep_cpu[i//width,i%width])]
        pruned.extend(t*hw+i for i in range(hw) if not bool(keep_cpu[i//width,i%width]))
        parent={i:i for i in selected}
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]]; i=parent[i]
            return i
        def union(a,b):
            a,b=find(a),find(b)
            if a!=b: parent[max(a,b)]=min(a,b)
        # Four undirected edge directions cover all eight neighbors, without
        # connecting across discarded patches or a radius-2 static gap.
        for dy,dx in ((0,1),(1,-1),(1,0),(1,1)):
            y0,y1=max(0,-dy),min(height,height-dy)
            x0,x1=max(0,-dx),min(width,width-dx)
            if y1<=y0 or x1<=x0: continue
            a=normalized[t,y0:y1,x0:x1]
            b=normalized[t,y0+dy:y1+dy,x0+dx:x1+dx]
            valid=(keep[y0:y1,x0:x1]&keep[y0+dy:y1+dy,x0+dx:x1+dx]
                   &((a*b).sum(-1)>object_threshold)).detach().cpu()
            for yr,xr in valid.nonzero().tolist():
                y,x=y0+yr,x0+xr
                union(y*width+x,(y+dy)*width+x+dx)
        connected={}
        for i in selected: connected.setdefault(find(i),[]).append(t*hw+i)
        for members in sorted(connected.values(),key=lambda g:g[0]):
            component_id=len(components)
            chosen,subgroups=_dpc_groups(flat,members)
            if stage_mode=='pruner_only':chosen=list(members);subgroups=[[i] for i in members]
            components.append(dict(component_id=component_id,time_group=t,indices=members,centers=chosen))
            for center,group in zip(chosen,subgroups):
                centers.append(center); groups.append(group); time_groups.append(t)
                kinds.append('component'); component_ids.append(component_id)
    minimum=scene_count+len(components);maximum=scene_count+sum(min(2,len(c['indices'])) for c in components)
    if stage_mode=='pruner_only':minimum=maximum=len(groups)
    if target_tokens is not None:
        if not minimum<=target_tokens<=maximum:
            raise ValueError(f'infeasible exact budget {target_tokens}; allowed [{minimum},{maximum}] with no background refill')
        if stage_mode!='pruner_only':
            extra=target_tokens-minimum
            doubled=set(c['component_id'] for c in sorted(components,key=lambda c:(-len(c['indices']),c['component_id'])) if len(c['indices'])>1)
            doubled=set(sorted(doubled,key=lambda i:(-len(components[i]['indices']),i))[:extra])
            centers=centers[:scene_count];groups=groups[:scene_count];time_groups=time_groups[:scene_count]
            kinds=kinds[:scene_count];component_ids=component_ids[:scene_count]
            for component in components:
                identity=component['component_id'];chosen,subgroups=_dpc_groups(flat,component['indices'],2 if identity in doubled else 1)
                component['centers']=chosen
                for center,group in zip(chosen,subgroups):
                    centers.append(center);groups.append(group);time_groups.append(component['time_group'])
                    kinds.append('component');component_ids.append(identity)
    n=tcount*hw
    return dict(schema='scene_connected_dpc_plan_v1',grid_thw=[tcount,height,width],
        original_tokens=n,output_tokens=len(groups),actual_retention=len(groups)/n,
        scene_tokens=scene_count,groups=groups,centers=centers,time_groups=time_groups,
        kinds=kinds,component_ids=component_ids,components=components,pruned_indices=pruned,
        background_threshold=background_threshold,object_threshold=object_threshold,
        stage_mode=stage_mode,scene_mode=scene_mode,requested_tokens=target_tokens,feasible_token_interval=[minimum,maximum],
        entity_identity_verified=False)


def apply_scene_plan(main,deepstack,positions,plan):
    n=plan['original_tokens']
    streams=[main]+list(deepstack)
    if len(deepstack)!=3 or any(not isinstance(x,torch.Tensor) or x.ndim!=2 or len(x)!=n
            or not x.is_floating_point() or x.device!=main.device or not torch.isfinite(x).all().item()
            for x in streams):
        raise ValueError('main and exactly three aligned finite DeepStack tensors required')
    if not isinstance(positions,torch.Tensor) or list(positions.shape)!=[3,n]:
        raise ValueError('original three-axis positions required')
    covered=[i for g in plan['groups'] for i in g]+plan['pruned_indices']
    if sorted(covered)!=list(range(n)) or len(set(covered))!=len(covered):
        raise ValueError('plan must partition all original tokens without duplicates')
    if len(plan['centers'])!=len(plan['groups']): raise ValueError('center/group mismatch')
    hw=plan['grid_thw'][1]*plan['grid_thw'][2]
    for center,group in zip(plan['centers'],plan['groups']):
        if center not in group or len({i//hw for i in group})!=1:
            raise ValueError('invalid center or cross-time merge')
    def aggregate(stream):
        return torch.stack([stream[group].float().mean(0).to(stream.dtype) for group in plan['groups']])
    return dict(main=aggregate(main),deepstack=[aggregate(x) for x in deepstack],
        positions=positions[:,plan['centers']],token_to_original_patch_indices=plan['groups'])
