from ultralytics import YOLO
import cv2
import json
from pathlib import Path

# 加载模型
det_model = YOLO('runs/detect/strawberry_stem_detection/weights/best.pt')  # 检测模型（stem + strawberry）
cls_model = YOLO('runs/classify/strawberry_ripeness/weights/best.pt')      # 分类模型（ripe/unripe/overripe）

# 颜色定义
COLORS = {
    'ripe': (0, 255, 0),       # 绿色 - 采摘
    'overripe': (0, 165, 255), # 橙色 - 记录
    'unripe': (128, 128, 128), # 灰色 - 不采摘
    'stem': (255, 255, 0),     # 黄色
    'pick': (0, 255, 255),     # 黄色十字
}


def classify_strawberry(img, box):
    """裁剪草莓区域并分类成熟度"""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 'unripe', 0.0
    results = cls_model.predict(source=crop, verbose=False)
    probs = results[0].probs
    cls_id = int(probs.top1)
    conf = float(probs.top1conf)
    cls_name = results[0].names[cls_id]
    # 低置信度的 unripe 判断为 ripe
    if cls_name == 'unripe' and conf < 0.6:
        return 'ripe', conf
    return cls_name, conf


def find_nearest_stem(strawberry_box, stems):
    """找到离草莓最近的茎"""
    sb = strawberry_box
    s_cx = (sb[0] + sb[2]) / 2
    s_cy = (sb[1] + sb[3]) / 2

    best_stem = None
    min_dist = float('inf')
    for stem in stems:
        b = stem['box']
        st_cx = (b[0] + b[2]) / 2
        st_cy = (b[1] + b[3]) / 2
        dist = ((s_cx - st_cx)**2 + (s_cy - st_cy)**2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_stem = stem
    return best_stem, min_dist


def get_pick_point(stem_box, offset_ratio=0.3):
    """根据茎计算采摘点"""
    b = stem_box
    pick_x = int((b[0] + b[2]) / 2)
    pick_y = int(b[3] - (b[3] - b[1]) * offset_ratio)
    return pick_x, pick_y


def process_image(image_path, save_dir='harvest_results'):
    """处理单张图像：检测 -> 分类 -> 采摘决策"""
    img = cv2.imread(str(image_path))
    det_results = det_model.predict(source=image_path, conf=0.1, verbose=False)

    # 提取检测结果
    stems = []
    strawberries = []
    for result in det_results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0:
                stems.append({'box': [x1, y1, x2, y2], 'conf': conf})
            else:
                strawberries.append({'box': [x1, y1, x2, y2], 'conf': conf})

    pick_points = []
    overripe_list = []

    for strawberry in strawberries:
        ripeness, ripe_conf = classify_strawberry(img, strawberry['box'])
        b = strawberry['box']
        color = COLORS.get(ripeness, (128, 128, 128))

        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
        cv2.putText(img, f"{ripeness} {ripe_conf:.2f}", (int(b[0]), int(b[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        nearest_stem, dist = find_nearest_stem(b, stems)

        if ripeness == 'ripe' and nearest_stem is not None:
            pick_x, pick_y = get_pick_point(nearest_stem['box'])
            pick_points.append({
                'pixel': (pick_x, pick_y),
                'ripeness': ripeness,
                'confidence': ripe_conf
            })
            cv2.drawMarker(img, (pick_x, pick_y), COLORS['pick'], cv2.MARKER_CROSS, 20, 3)
            cv2.putText(img, "PICK", (pick_x + 10, pick_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['pick'], 2)
            sb = nearest_stem['box']
            cv2.rectangle(img, (int(sb[0]), int(sb[1])), (int(sb[2]), int(sb[3])), COLORS['stem'], 2)

        elif ripeness == 'overripe':
            overripe_list.append({
                'bbox': [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                'confidence': ripe_conf,
                'center': (int((b[0]+b[2])/2), int((b[1]+b[3])/2))
            })
            cv2.putText(img, "OVERRIPE - LOGGED", (int(b[0]), int(b[3]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['overripe'], 2)

        # unripe: 只显示框，不采摘

    Path(save_dir).mkdir(exist_ok=True)
    output_path = Path(save_dir) / Path(image_path).name
    cv2.imwrite(str(output_path), img)

    return pick_points, overripe_list, output_path


# 处理测试集图像
test_images_dir = Path('dataset/test/images')
image_files = [f for ext in ('*.jpg', '*.png') for f in test_images_dir.glob(ext)]

print(f"处理 {len(image_files)} 张图像...\n")

all_pick_points = []
all_overripe = []

for img_path in image_files:
    pick_points, overripe_list, output_path = process_image(img_path)

    print(f"图像: {img_path.name}")
    if pick_points:
        print(f"  [RIPE] 采摘点: {len(pick_points)} 个")
        for i, p in enumerate(pick_points, 1):
            print(f"    #{i}: ({p['pixel'][0]}, {p['pixel'][1]}) conf={p['confidence']:.2f}")
    if overripe_list:
        print(f"  [OVERRIPE] 记录: {len(overripe_list)} 个")
    print(f"  结果保存: {output_path}\n")
    all_pick_points.extend([{
        'image': img_path.name,
        **p
    } for p in pick_points])

    all_overripe.extend([{
        'image': img_path.name,
        **o
    } for o in overripe_list])

# 保存 overripe JSON
overripe_json_path = Path('harvest_results') / 'overripe_strawberries.json'
with open(overripe_json_path, 'w', encoding='utf-8') as f:
    json.dump(all_overripe, f, indent=2, ensure_ascii=False)

print(f"\n===== 汇总 =====")
print(f"采摘 (ripe): {len(all_pick_points)} 个")
print(f"记录 (overripe): {len(all_overripe)} 个 -> {overripe_json_path}")
print(f"图像结果: harvest_results/")
