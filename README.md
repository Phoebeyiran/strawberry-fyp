# 草莓和茎检测 - YOLOv8 训练

## 项目结构

```
strawberry-work/
├── dataset/                    # 数据集
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── train_strawberry.py         # 训练脚本
├── predict_strawberry.py       # 推理脚本
└── runs/                       # 训练结果（自动生成）
```

## 使用步骤

### 1. 激活环境

```bash
conda activate strawberry
```

### 2. 安装依赖（如果还没安装）

```bash
pip install ultralytics pillow
```

### 3. 训练模型

```bash
cd ~/strawberry-work
python train_strawberry.py
```

训练参数：
- epochs: 100
- batch: 16
- imgsz: 640
- device: mps (Mac GPU)

### 4. 推理测试

```bash
python predict_strawberry.py
```

## 结果位置

- 训练权重: `runs/detect/strawberry_stem_detection/weights/best.pt`
- 训练日志和图表: `runs/detect/strawberry_stem_detection/`
- 推理结果: `runs/detect/strawberry_predictions/`

## 检测类别

- 0: stem (茎)
- 1: strawberry (草莓)
