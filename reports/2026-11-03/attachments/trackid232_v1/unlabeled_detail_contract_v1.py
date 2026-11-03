"""I2: same native crop content and order, without object-ID display or catalog."""
import hashlib
import math
from pathlib import Path
import re


def crop_plan(details):
    fields = ['native_crop_file', 'native_crop_sha256', 'crop_rgb_sha256',
              'crop_size', 'f', 'native_index', 'time_s']
    plan = []
    for detail in details:
        filename = detail['native_crop_file']
        if not isinstance(filename, str) or not re.fullmatch(r'[A-Za-z0-9_-]+\.png', filename):
            raise ValueError('Native crop filename outside source directory')
        plan.append({name: detail[name] for name in fields})
    return plan


def build_prompt(original, details):
    if not isinstance(original, str) or not original.strip():
        raise ValueError('Missing original prompt')
    # Do not expose hidden detector IDs, track IDs, class suggestions, or filenames.
    return original + '\n\n' + '''补充图片是同一原始MP4在已提供F帧上的原生分辨率局部截图，帮助看清对象外观和手部细节。完整视频仍包含全部原10FPS抽帧加末帧，动作与相互关系以完整视频为准。
局部图只标来源F帧、原生帧索引和媒体时间，不给对象编号。同一人可能在多帧局部图里反复出现，不按截图数量推断人数；检测框可能包含旁人、背景、误检或漏检，出现局部图不代表异常。未被局部图覆盖的真实异常仍需检查完整视频。
仍按原JSON字段用可见外貌、动作、携物及交互关系指认对象，object_id由你按实际连续身份命名，不选图片编号。不能仅因人物衣着相似或图页相邻就认定同一身份，不编造不可见细节。'''


def render_pages(details, source_folder, target_folder, *, font):
    from PIL import Image, ImageDraw, ImageOps
    source_folder, target_folder = Path(source_folder), Path(target_folder)
    plan = crop_plan(details)
    crops = []
    for item in plan:
        path = source_folder / item['native_crop_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['native_crop_sha256']:
            raise ValueError('Original native PNG differs')
        with Image.open(path) as raw:
            image = raw.convert('RGB')
        if list(image.size) != item['crop_size'] or hashlib.sha256(image.tobytes()).hexdigest() != item['crop_rgb_sha256']:
            raise ValueError('Original native crop pixels differ')
        crops.append(image)
    target_folder.mkdir(parents=True, exist_ok=True)
    galleries = []
    for start in range(0, len(plan), 6):
        selected = plan[start:start + 6]
        page = Image.new('RGB', (512, 288 * math.ceil(len(selected) / 2)), (24, 24, 24))
        draw = ImageDraw.Draw(page)
        for k, item in enumerate(selected):
            row, col = divmod(k, 2)
            image = ImageOps.contain(crops[start + k], (252, 240))
            page.paste(image, (col * 256 + (256 - image.width) // 2,
                              row * 288 + 45 + (240 - image.height) // 2))
            draw.text((col * 256 + 3, row * 288 + 2), f"F{item['f']:03d}", font=font, fill='yellow')
            draw.text((col * 256 + 3, row * 288 + 23),
                      f"native={item['native_index']} t={item['time_s']:.2f}s", font=font, fill='white')
        name = f'native_views_{len(galleries) + 1:02d}.jpg'
        path = target_folder / name
        if path.exists():
            raise ValueError('Existing I2 gallery; never overwrite')
        page.save(path, quality=96)
        galleries.append(name)
    return dict(galleries=galleries, source_crop_count=len(plan),
                source_crop_rgb_sha256=[x['crop_rgb_sha256'] for x in plan])
