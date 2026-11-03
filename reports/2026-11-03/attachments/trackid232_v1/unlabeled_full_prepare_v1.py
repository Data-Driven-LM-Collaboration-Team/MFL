"""Server CPU source inventory and native crops for the fixed full cohort."""
import hashlib
from pathlib import Path
import shutil
import sys

from track_contract import read, save, sha, now
from native_detail_contract_v1 import choose_details, project_crop
from unlabeled_detail_contract_v1 import render_pages

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'unlabeled_full_inputs_v1'


def verify():
    freeze = read(BASE / 'UNLABELED_FULL_FREEZE_V1.json')
    for name, digest in freeze['files'].items():
        assert sha(BASE / name) == digest, name
    return freeze


def inventory():
    import cv2
    code = read(BASE / 'UNLABELED_FULL_CODE_FREEZE_V1.json')
    for name, digest in code['files'].items():
        assert sha(BASE / name) == digest, name
    parent = BASE / 'UNLABELED_DETAIL_FREEZE_V1.json'
    assert sha(parent) == code['parent_I2_freeze_sha256']
    auth = read(BASE / 'UNLABELED_FULL_EXPANSION_AUTH_V1.json')
    assert auth['predeclared_gate_passed'] is True
    rows = read(BASE / 'inference_manifest.json')
    assert len(rows) == 232 and len({x['clip_id'] for x in rows}) == 232
    old_sources = {x['clip_id']: x for x in read(BASE / 'NATIVE_DETAIL_SOURCES_V1.json')['rows']}
    cfg = read(BASE / 'unlabeled_full_config_v1.json')
    pilot = read(BASE / 'candidate_config.json')['engineering_ordinals']
    sources = []
    parents = {}
    cv2.setNumThreads(2)
    for ordinal, row in enumerate(rows, 1):
        cid = row['clip_id']
        native = Path('/workspace/VAD-V2/runs/qwen3vl8b_anomaly232_20261021/media') / (cid + '.mp4')
        tracks = BASE / 'candidates_v1' / 'candidates' / cid / 'TRACKS.json'
        candidate = BASE / 'candidates_v1' / 'candidate_records' / (cid + '.json')
        media = Path(cfg['source_root']) / 'media' / cid / 'MEDIA.json'
        tr = read(tracks); cand = read(candidate); med = read(media)
        assert cand['state'] == 'prepared' and sha(tracks) == cand['tracks_sha256']
        assert sha(native) == row['video_sha256'] == med['source_sha256']
        assert tr['native_indices'] == med['native_indices'] == cand['source_native_indices']
        assert tr['fps'] == row['fps'] and med['source_sample_fps'] == 10
        cap = cv2.VideoCapture(str(native))
        try:
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            cap.release()
        assert size == (row['width'], row['height'])
        source = dict(ordinal=ordinal, clip_id=cid, native_video_path=str(native),
            native_video_sha256=row['video_sha256'], native_width=size[0], native_height=size[1],
            tracks_path=str(tracks), tracks_sha256=sha(tracks), candidate_record_path=str(candidate),
            candidate_record_sha256=sha(candidate), media_path=str(media), media_sha256=sha(media),
            detector_width=med['width'], detector_height=med['height'], track_count=len(tr['tracks']))
        if ordinal in pilot:
            assert source == old_sources[cid], cid
            record = BASE / 'unlabeled_detail_arm_v1' / 'records' / (cid + '.json')
            terminal = read(record)
            assert terminal['freeze_sha256'] == code['parent_I2_freeze_sha256']
            assert terminal['clip_id'] == cid and terminal['ordinal'] == ordinal
            folder = BASE / 'unlabeled_detail_arm_v1' / 'clips' / cid
            parents[cid] = dict(ordinal=ordinal, record_sha256=sha(record),
                files_sha256={p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
        sources.append(source)
        print('source_verified', ordinal, cid, flush=True)
    source_path = BASE / 'UNLABELED_FULL_SOURCES_V1.json'
    value = dict(rows=sources, inherited_terminals=parents, reference_answers_used=False,
                 human_verified=False, eligible_for_formal_supervised_training=False)
    if source_path.exists():
        assert read(source_path) == value
    else:
        save(source_path, value)
    files = dict(code['files']); files[source_path.name] = sha(source_path)
    value = dict(at_utc=now(), arm='I3_full232_same_I2_method', files=files,
        code_freeze_sha256=sha(BASE / 'UNLABELED_FULL_CODE_FREEZE_V1.json'),
        source_inventory_sha256=sha(source_path), expected=232, inherited_n=12, new_calls=220,
        parent_I2_freeze_sha256=code['parent_I2_freeze_sha256'], inherited_terminals=parents,
        reference_answers_used=False, human_verified=False, eligible_for_formal_supervised_training=False)
    path = BASE / 'UNLABELED_FULL_FREEZE_V1.json'
    if path.exists():
        old = read(path)
        for key in value:
            if key != 'at_utc':
                assert value[key] == old[key], key
    else:
        save(path, value)


def main():
    import cv2
    from PIL import Image, ImageFont
    freeze = verify(); cfg = read(BASE / 'unlabeled_full_config_v1.json')
    fingerprint = sha(BASE / 'UNLABELED_FULL_FREEZE_V1.json')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    cv2.setNumThreads(2)
    completed = []
    for source in read(BASE / 'UNLABELED_FULL_SOURCES_V1.json')['rows']:
        cid = source['clip_id']; folder = ROOT / 'clips' / cid
        terminal = ROOT / 'records' / (cid + '.json')
        if terminal.exists():
            record = read(terminal)
            assert record['freeze_sha256'] == fingerprint and record['state'] == 'prepared'
            assert sha(folder / 'DETAILS.json') == record['details_sha256']
            for name, meta in record['images'].items():
                assert sha(folder / name) == meta['sha256']
            completed.append(record); continue
        folder.mkdir(parents=True, exist_ok=True)
        if cid in freeze['inherited_terminals']:
            original = BASE / 'unlabeled_details_v1' / 'clips' / cid
            parent_path = BASE / 'unlabeled_details_v1' / 'records' / (cid + '.json')
            parent = read(parent_path)
            assert parent['freeze_sha256'] == freeze['parent_I2_freeze_sha256']
            for name in list(parent['galleries']) + ['DETAILS.json']:
                target = folder / name
                if target.exists():
                    assert sha(target) == sha(original / name)
                else:
                    shutil.copyfile(original / name, target)
            assert sha(folder / 'DETAILS.json') == parent['details_sha256']
            for name, meta in parent['images'].items():
                assert sha(folder / name) == meta['sha256']
            record = dict(parent, freeze_sha256=fingerprint,
                          inherited_preparation_record_sha256=sha(parent_path))
        else:
            for prefix in ['native_video', 'tracks', 'candidate_record', 'media']:
                assert sha(source[prefix + '_path']) == source[prefix + '_sha256']
            tracks = read(source['tracks_path'])
            details = choose_details(tracks['tracks'], source['detector_width'], source['detector_height'], cfg['detail_max_tracks'])
            assert len(details) <= 3 * cfg['detail_max_tracks']
            for d in details:
                assert tracks['native_indices'][d['f']] == d['native_index']
            requested = {d['native_index'] for d in details}; frames = {}
            cap = cv2.VideoCapture(source['native_video_path'])
            try:
                for index in range(max(requested, default=-1) + 1):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError('Native decode ended before requested frame')
                    if index in requested:
                        assert (frame.shape[1], frame.shape[0]) == (source['native_width'], source['native_height'])
                        frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            finally:
                cap.release()
            assert set(frames) == requested
            for d in details:
                frame = frames[d['native_index']]
                box = project_crop(d['box'], (source['detector_width'], source['detector_height']),
                    (source['native_width'], source['native_height']), cfg['detail_pad_ratio'])
                crop = Image.fromarray(frame).crop(box); name = d['detail_id'] + '.png'
                crop.save(folder / name)
                d.update(native_crop_box_xyxy=box, native_frame_rgb_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                    crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(), crop_size=list(crop.size),
                    native_crop_file=name, native_crop_sha256=sha(folder / name))
                crop.close()
            result = render_pages(details, folder, folder, font=font)
            save(folder / 'DETAILS.json', dict(clip_id=cid, details=details, source=source,
                object_reference_ids_visible_to_model=False, reference_answers_used=False,
                human_verified=False, eligible_for_formal_supervised_training=False))
            record = dict(clip_id=cid, ordinal=source['ordinal'], state='prepared', at_utc=now(),
                freeze_sha256=fingerprint, source_native_indices=tracks['native_indices'],
                native_video_sha256=source['native_video_sha256'], details_sha256=sha(folder / 'DETAILS.json'),
                galleries=result['galleries'], images={name: dict(sha256=sha(folder / name), size=(folder / name).stat().st_size)
                    for name in result['galleries']}, detail_count=len(details),
                source_crop_rgb_sha256=[d['crop_rgb_sha256'] for d in details],
                reference_answers_used=False, previous_predictions_used=False,
                human_verified=False, eligible_for_formal_supervised_training=False)
        assert not terminal.exists()
        save(terminal, record); completed.append(record)
        save(ROOT / 'STATUS.json', dict(stage='preparing', prepared=len(completed), expected=232, at_utc=now()))
        print('prepared', source['ordinal'], cid, record['detail_count'], flush=True)
    assert len(completed) == 232
    save(ROOT / 'STATUS.json', dict(stage='prepared', prepared=232, expected=232,
        details=sum(r['detail_count'] for r in completed), model_calls=0, gpu_allocations=0, at_utc=now()))


if __name__ == '__main__':
    inventory() if '--inventory' in sys.argv else main()
