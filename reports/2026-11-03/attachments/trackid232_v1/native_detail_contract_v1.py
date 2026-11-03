"""Detector-only detail selection; a reference is one box in one native frame."""
import copy
import math


def choose_details(tracks, width, height, max_tracks=12):
    if width <= 0 or height <= 0 or max_tracks < 1:
        raise ValueError('Invalid detail selection bounds')
    ranked = []
    for track in tracks:
        observations = track['observations']
        if not observations:
            raise ValueError('Empty candidate track')
        kind = track['kind']
        if kind == 'person':
            priority = 0
        elif kind in ['animal', 'dog', 'cat', 'horse', 'backpack', 'handbag', 'suitcase']:
            priority = 1
        elif kind in ['car', 'bicycle', 'motorcycle', 'bus', 'truck']:
            centers = [((x['box'][0] + x['box'][2]) / 2, (x['box'][1] + x['box'][3]) / 2) for x in observations]
            displacement = math.hypot(max(x[0] for x in centers) - min(x[0] for x in centers),
                                      max(x[1] for x in centers) - min(x[1] for x in centers))
            if displacement < .03 * math.hypot(width, height):
                continue
            priority = 2
        else:
            continue
        ranked.append((priority, -len(observations), -max(x['score'] for x in observations), track['track_id'], track))
    selected = sorted(ranked, key=lambda x: x[:4])[:max_tracks]
    details, seen = [], set()
    for _, _, _, _, track in selected:
        observations = sorted(track['observations'], key=lambda x: x['f'])
        for index in sorted({0, len(observations) // 2, len(observations) - 1}):
            observation = observations[index]
            key = (observation['native_index'], tuple(observation['box']))
            if key in seen:
                continue
            seen.add(key)
            detail = copy.deepcopy(observation)
            detail.update(detail_id=f'D{len(details) + 1:03d}', source_track_id=track['track_id'],
                          identity_scope='one_object_in_one_native_frame')
            details.append(detail)
    return details


def project_crop(box, detector_size, native_size, pad_ratio=.12):
    dw, dh = detector_size
    nw, nh = native_size
    if min(dw, dh, nw, nh) <= 0 or len(box) != 4 or not all(math.isfinite(x) for x in box):
        raise ValueError('Invalid crop coordinate system')
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1 or pad_ratio < 0:
        raise ValueError('Invalid crop box')
    x1, x2 = x1 * nw / dw, x2 * nw / dw
    y1, y2 = y1 * nh / dh, y2 * nh / dh
    pad = pad_ratio * max(x2 - x1, y2 - y1)
    value = [max(0, math.floor(x1 - pad)), max(0, math.floor(y1 - pad)),
             min(nw, math.ceil(x2 + pad)), min(nh, math.ceil(y2 + pad))]
    if value[2] <= value[0] or value[3] <= value[1]:
        raise ValueError('Crop outside native frame')
    return value


def build_prompt(original, details):
    if not isinstance(original, str) or not original.strip():
        raise ValueError('Missing original prompt')
    listing = '\n'.join(f"{d['detail_id']} F{d['f']:03d} t={d['time_s']:.3f}s" for d in details)
    return original + '\n\n' + '''补充图片是同一原始MP4在已提供F帧上的原生分辨率局部截图，帮助看清对象外观和手部细节。完整视频仍包含全部原10FPS抽帧加末帧，动作与相互关系以完整视频为准。
每个D编号只指向该F帧中的一个检测框，不是跨帧永久身份；不同D可能是同一个人，同一检测轨迹也可能换人，不能按截图数推断人数。候选可能误检、漏检或没有异常，出现截图不代表异常。截图未覆盖的真实异常仍按原任务用可见语言描述，不强选编号。
仍按原JSON字段作答，不要求D编号；如果引用编号，必须同时给出可见外貌或动作。不要把本补充目录当成事件或政策答案。
局部图来源帧：
''' + (listing or '本片没有可用的局部候选图，仍检查完整视频。')
