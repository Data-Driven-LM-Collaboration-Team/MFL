"""Native multimodal bridge; original non-video tokens and 3-axis positions."""
from pathlib import Path
import time,hashlib
import numpy as np
import torch
from prune_merge_core import build_plan,apply_plan
from sequence_adapter import native_kwargs

METHODS={'full_native':None,'select70':'select','merge70':'merge','prune_merge70':'prune_merge','norm_merge70':'merge','random_merge70':'merge','pm90':'prune_merge','pm92':'prune_merge','merge92':'merge','select90':'select','global90':'select','anchor90':'select'}

@torch.inference_mode()
def prepare_prune_merge_sequence(model,inputs,method,retention=.30,*,artifact_path=None):
    if method not in METHODS:raise ValueError('unregistered compressor')
    device=inputs['input_ids'].device
    def tick():
        if device.type=='cuda':torch.cuda.synchronize(device)
        return time.perf_counter()
    start=tick();grid=inputs['video_grid_thw']
    if grid.shape!=(1,3) or model.config.vision_config.spatial_merge_size!=2 or model.config.vision_config.temporal_patch_size!=2:raise ValueError('one native temporal2 spatial2 video required')
    t,h,w=grid[0].tolist();h//=2;w//=2
    image=None
    if inputs.get('pixel_values') is not None:image=model.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'],return_dict=True)
    video=model.get_video_features(inputs['pixel_values_videos'],grid,return_dict=True)
    main=torch.cat(video.pooler_output);deep=[torch.cat(x) for x in video.deepstack_features]
    encoded=tick();ids=inputs['input_ids'];types=inputs['mm_token_type_ids']
    vmask=ids==model.config.video_token_id;imask=ids==model.config.image_token_id
    if ids.shape[0]!=1 or not torch.equal(vmask,types==2) or not torch.equal(imask,types==1) or not bool((inputs['attention_mask']==1).all()):raise ValueError('modality/alignment mismatch')
    n=int(vmask.sum());ni=int(imask.sum())
    if n!=len(main) or len(main)!=t*h*w:raise ValueError('feature count differs')
    positions,_=model.model.get_rope_index(ids,image_grid_thw=inputs.get('image_grid_thw'),video_grid_thw=grid,attention_mask=inputs['attention_mask'],mm_token_type_ids=types)
    if method=='full_native':
        # No identity groups or scatter: preserve original encoder feature objects.
        length=ids.shape[1];aligned={k:inputs[k] for k in ('input_ids','attention_mask','mm_token_type_ids')}
        aligned.update(position_ids=torch.cat([torch.arange(length,device=device,dtype=positions.dtype).reshape(1,1,-1),positions]),
            rope_deltas=(positions.max()+1-length).reshape(1,1),main=main,deepstack=deep)
        planned=aggregated=tick();plan=None;kept=n
    else:
        signal='norm' if method=='norm_merge70' else ('random' if method=='random_merge70' else 'novelty')
        if method in ('select90','global90','anchor90'):
            from purity_core import build_purity_plan
            plan=build_purity_plan(main.reshape(t,h,w,-1),image_tokens=ni,retention=retention,variant=method)
        elif method in ('pm90','pm92','merge92'):
            from high_compression_core import build_high_plan
            plan=build_high_plan(main.reshape(t,h,w,-1),image_tokens=ni,retention=retention,mode=METHODS[method])
        else:plan=build_plan(main.reshape(t,h,w,-1),image_tokens=ni,retention=retention,mode=METHODS[method],signal=signal)
        planned=tick();merged,merged_deep=apply_plan(main,deep,plan)
        physical=torch.where(vmask[0])[0];keep=~vmask[0];keep[physical[plan['centers']]]=True
        length=int(keep.sum());kept=len(plan['centers'])
        aligned={k:inputs[k][:,keep] for k in ('input_ids','attention_mask','mm_token_type_ids')}
        aligned.update(position_ids=torch.cat([torch.arange(length,device=device,dtype=positions.dtype).reshape(1,1,-1),positions[:,:,keep]]),
            rope_deltas=(positions.max()+1-length).reshape(1,1),main=merged,deepstack=merged_deep)
        if not torch.equal(ids[~vmask],aligned['input_ids'][aligned['mm_token_type_ids']!=2]):raise ValueError('non-video tokens changed')
        aggregated=tick()
    prepared=native_kwargs(model,aligned,image);artifact=None
    summary=None
    if plan is not None:
        summary={k:v for k,v in plan.items() if not isinstance(v,torch.Tensor)}
        summary.update(retained=int((plan['assignment']>=0).sum()),pruned=int((plan['assignment']<0).sum()),output_tokens=kept)
        if artifact_path is not None:
            path=Path(artifact_path)
            arrays={k:v.detach().cpu().numpy() for k,v in plan.items() if isinstance(v,torch.Tensor)}
            arrays['grid_thw']=np.asarray(plan['grid_thw'])
            with path.open('xb') as f:np.savez_compressed(f,**arrays)
            artifact=dict(filename=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    end=tick()
    return prepared,dict(video_tokens_before=n,video_tokens_after=kept,image_tokens_before=ni,image_tokens_after=ni,
        input_tokens_before=ids.shape[1],input_tokens_after=length,actual_visual_retention=(kept+ni)/(n+ni),
        all_video_frames_encoded=True,all_image_features_unchanged=True,non_video_tokens_all_retained=True,
        encoder_cost_reduced=False,entity_binding_enabled=False,plan_summary=summary,plan_artifact=artifact,
        native_full_no_identity_reconstruction=method=='full_native',encoder_s=encoded-start,
        planning_and_position_s=planned-encoded,aggregation_alignment_s=aggregated-planned,
        plan_export_and_receipt_s=end-aggregated)
