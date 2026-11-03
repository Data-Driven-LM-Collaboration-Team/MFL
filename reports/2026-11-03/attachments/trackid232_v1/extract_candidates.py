"""Server-only, inference-input candidate preparation. No reference labels."""
from pathlib import Path
import argparse
import csv
import io
import os
import subprocess
import sys
import time
import traceback
from track_contract import read,save,sha,now,collect_tracks,render_gallery

ROOT=Path(__file__).resolve().parent


def verify_candidate_freeze():
    freeze=read(ROOT/'CANDIDATE_FREEZE_V0.json')
    for relative,digest in freeze['files'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Candidate frozen file differs: '+relative)


def worker(stage):
    import cv2
    import numpy as np
    import torch
    from PIL import Image,ImageDraw,ImageFont
    from ultralytics import YOLO
    verify_candidate_freeze()
    cfg=read(ROOT/'candidate_config.json');ready=read(ROOT/'DETECTOR_MODEL_READY.json')
    if sha(ROOT/'detector/yolo11s.pt')!=ready['sha256']:raise ValueError('Detector weight hash mismatch')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_cap_mib']*1024**2/torch.cuda.get_device_properties(0).total_memory)
    sys.path.insert(0,cfg['source_root']);import run_8b as legacy
    model=YOLO(str(ROOT/'detector/yolo11s.pt'))
    rows=read(ROOT/'inference_manifest.json')
    if stage=='preflight_v0':rows=[r for i,r in enumerate(rows,1) if i in cfg['engineering_ordinals']]
    out=ROOT/stage;out.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    for ordinal,row in enumerate(rows,1):
        cid=row['clip_id'];folder=out/'candidates'/cid;terminal=out/'candidate_records'/(cid+'.json')
        if terminal.exists():continue
        start=time.monotonic()
        try:
            save(out/'CANDIDATE_STATUS.json',dict(stage='extracting',clip_id=cid,ordinal=ordinal,expected=len(rows),at_utc=now(),pid=os.getpid()))
            frames,receipt=legacy.decode(row);rgb=frames.permute(0,2,3,1).numpy()
            if getattr(model,'predictor',None) is not None:
                for tracker in getattr(model.predictor,'trackers',[]):tracker.reset()
            detections=[];shot=0;previous=None
            for f,image in enumerate(rgb):
                thumbnail=cv2.resize(image,(32,32)).astype(np.float32)
                delta=None if previous is None else float(np.abs(thumbnail-previous).mean())
                if delta is not None and delta>cfg['cut_mean_absolute_rgb_threshold']:
                    shot+=1
                    for tracker in getattr(model.predictor,'trackers',[]):tracker.reset()
                previous=thumbnail
                result=model.track(image[...,::-1].copy(),persist=True,tracker=str(ROOT/'bytetrack.yaml'),
                    device=0,imgsz=cfg['imgsz'],half=True,conf=cfg['detector_conf'],iou=cfg['detector_iou'],
                    classes=cfg['classes'],max_det=100,verbose=False,save=False)[0]
                boxes=[]
                if result.boxes is not None:
                    b=result.boxes.cpu();ids=[None]*len(b) if b.id is None else b.id.int().tolist()
                    for xyxy,score,cls,raw in zip(b.xyxy.tolist(),b.conf.tolist(),b.cls.int().tolist(),ids):
                        boxes.append(dict(raw_id=raw,box=xyxy,score=score,kind=result.names[cls]))
                detections.append(dict(f=f,shot=shot,cut_delta=delta,boxes=boxes))
            all_tracks=collect_tracks(detections,receipt['native_indices'],row['fps'])
            selected=[t for t in all_tracks if len(t['observations'])>=2 or max(o['score'] for o in t['observations'])>=.4]
            galleries=render_gallery(rgb,selected,folder)
            keys=[]
            for f in sorted({0,len(rgb)//2,len(rgb)-1}):
                image=Image.fromarray(rgb[f]);draw=ImageDraw.Draw(image)
                for t in selected:
                    for o in t['observations']:
                        if o['f']!=f:continue
                        box=o['box'];draw.rectangle(box,outline='orange',width=2)
                        x,y=box[:2];draw.rectangle((x,max(0,y-25),x+75,max(0,y-25)+24),fill='black')
                        draw.text((x+2,max(0,y-25)),t['track_id'],font=font,fill='yellow')
                name=f'keyframe_F{f:03d}.jpg';image.save(folder/name,quality=95);keys.append(name)
            save(folder/'RAW_DETECTIONS.json',dict(frames=detections,all_tracks=all_tracks))
            save(folder/'TRACKS.json',dict(clip_id=cid,tracks=selected,native_indices=receipt['native_indices'],fps=row['fps'],
                source_media_sha256=receipt['video_sha256'],candidate_config_sha256=sha(ROOT/'candidate_config.json'),
                detector_sha256=ready['sha256'],track_id_scope='clip_and_detected_shot_only',human_verified=False))
            images={name:dict(sha256=sha(folder/name),size=(folder/name).stat().st_size) for name in keys+galleries}
            record=dict(clip_id=cid,state='prepared',at_utc=now(),sampled_frames=len(rgb),source_native_indices=receipt['native_indices'],
                all_track_count=len(all_tracks),selected_track_count=len(selected),short_low_confidence_tracks_excluded=len(all_tracks)-len(selected),
                untracked_boxes=sum(b['raw_id'] is None for f in detections for b in f['boxes']),detected_shots=shot+1,
                galleries=galleries,keyframes=keys,images=images,tracks_sha256=sha(folder/'TRACKS.json'),
                code_sha256=sha(__file__),candidate_config_sha256=sha(ROOT/'candidate_config.json'),elapsed_s=time.monotonic()-start,
                actual_visual_quality='not_yet_adjudicated',human_verified=False,eligible_for_formal_supervised_training=False)
            save(terminal,record)
            print(cid,'prepared',len(selected),'tracks',len(rgb),'frames',flush=True)
            del rgb,frames;torch.cuda.empty_cache()
        except Exception:
            save(terminal,dict(clip_id=cid,state='candidate_preparation_failed',at_utc=now(),error=traceback.format_exc()))
            raise
    save(out/'CANDIDATE_STATUS.json',dict(stage='prepared_not_semantically_validated',processed=len(rows),expected=len(rows),at_utc=now()))


def queue(stage):
    import fcntl
    verify_candidate_freeze()
    cfg=read(ROOT/'candidate_config.json');out=ROOT/stage;out.mkdir(exist_ok=True)
    with (out/'candidate_queue.lock').open('a+') as single:
        try:fcntl.flock(single,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        deadline=time.monotonic()+6*3600
        while time.monotonic()<deadline:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            cards=[dict(index=int(r[0]),uuid=r[1].strip(),free_mib=int(r[2]),util=int(r[3])) for r in csv.reader(io.StringIO(text))]
            for card in sorted(cards,key=lambda c:c['free_mib'],reverse=True):
                if card['free_mib']<cfg['gpu_cap_mib']+cfg['gpu_headroom_mib'] or card['util']>20:continue
                leases=ROOT.parent.parent/'.gpu_leases';leases.mkdir(exist_ok=True)
                with (leases/(card['uuid']+'.lock')).open('a+') as lease:
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    latest=subprocess.check_output(['nvidia-smi','--id='+card['uuid'],'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
                    if int(latest[0])<cfg['gpu_cap_mib']+cfg['gpu_headroom_mib'] or int(latest[1])>20:continue
                    save(out/'CANDIDATE_GPU_LEASE.json',dict(**card,at_utc=now(),cooperative_lock_only=True))
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(ROOT/'ultralytics_settings'))
                    with (out/'candidate_runtime.log').open('ab') as log:
                        child=subprocess.run([str(ROOT/'detector_venv/bin/python'),'-u',str(ROOT/'extract_candidates.py'),'--worker','--stage',stage],cwd=ROOT,env=env,stdout=log,stderr=log)
                    save(out/'CANDIDATE_QUEUE_STATUS.json',dict(returncode=child.returncode,gpu_released=True,at_utc=now()))
                    return
            save(out/'CANDIDATE_STATUS.json',dict(stage='waiting_eligible_cooperative_gpu',at_utc=now()));time.sleep(30)
        raise TimeoutError('No eligible cooperative GPU within engineering queue window')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--stage',choices=['preflight_v0','candidates_v1'],required=True);args=parser.parse_args()
    worker(args.stage) if args.worker else queue(args.stage)
