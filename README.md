# Strawberry Harvesting Detection System

基于 YOLOv8 的草莓采摘检测系统，集成目标检测与成熟度分类，支持 Web 可视化界面查看结果。

## 项目结构

```
strawberry-work/
├── main.py                  # 主程序：检测 + 分类 + Web 可视化界面
├── harvest.py               # 采摘检测脚本（命令行版）
├── train_det.py             # 目标检测模型训练（stem + strawberry）
├── train_cls.py             # 成熟度分类模型训练（ripe / unripe / overripe）
├── config.yaml              # 配置文件（模型路径、置信度阈值、端口等）
├── requirements.txt         # Python 依赖
├── yolov8n.pt               # YOLOv8n 检测预训练权重
├── yolov8n-cls.pt           # YOLOv8n 分类预训练权重
├── dataset/                 # 数据集（Roboflow 导出）
│   ├── train/               #   训练集（images + labels）
│   ├── valid/               #   验证集
│   ├── test/                #   测试集
│   └── data.yaml            #   数据集配置
├── runs/                    # 训练结果（自动生成）
│   ├── detect/              #   检测模型训练输出
│   └── classify/            #   分类模型训练输出
└── harvest_results/         # 推理结果
    ├── *.jpg / *.png        #   标注后的图像
    ├── report.json          #   检测报告
    ├── overripe_strawberries.json  # 过熟草莓记录
    ├── index.html           #   Web 结果展示页面
    └── evaluation.html      #   模型评估页面
```

## 功能概述

1. **目标检测**：使用 YOLOv8 检测图像中的草莓（strawberry）和茎（stem）
2. **成熟度分类**：对检测到的草莓进行成熟度分类（ripe / unripe / overripe）
3. **采摘决策**：
   - **ripe**（成熟）：计算最近茎的采摘点，标记 PICK
   - **overripe**（过熟）：记录位置，不采摘
   - **unripe**（未熟）：仅显示边框，不采摘
4. **Web 可视化**：通过 FastAPI 提供网页界面，查看检测结果、统计数据和模型评估

## 安装

```bash
conda activate strawberry
pip install -r requirements.txt
```

依赖：
- ultralytics >= 8.0.0
- opencv-python-headless >= 4.8.0
- fastapi >= 0.100.0
- uvicorn >= 0.23.0
- pyyaml >= 6.0
- numpy >= 1.24.0

## 使用方法

### 1. 训练模型

**训练目标检测模型（stem + strawberry）：**

```bash
python train_det.py
# 可选参数: --epochs 100 --batch 16 --imgsz 640 --device mps
```

**训练成熟度分类模型（ripe / unripe / overripe）：**

```bash
python train_cls.py
# 可选参数: --epochs 50 --batch 32 --imgsz 224 --device mps
```

### 2. 运行主程序（检测 + Web 界面）

```bash
python main.py
```

启动后访问 http://localhost:8000 查看结果。可在 `config.yaml` 中修改端口和其他参数。

### 3. 命令行采摘检测

```bash
python harvest.py
```

处理 `dataset/test/images` 中的所有图像，结果保存至 `harvest_results/`。

## 配置

编辑 `config.yaml` 修改运行参数：

```yaml
det_model: runs/detect/strawberry_stem_detection/weights/best.pt
cls_model: runs/classify/strawberry_ripeness/weights/best.pt
det_conf: 0.1
cls_unripe_threshold: 0.6
source: dataset/test/images
save_dir: harvest_results
port: 8000
```

## 检测类别

| 模型 | 类别 | 说明 |
|------|------|------|
| 检测模型 | stem | 草莓茎 |
| 检测模型 | strawberry | 草莓果实 |
| 分类模型 | ripe | 成熟（可采摘） |
| 分类模型 | unripe | 未熟（不采摘） |
| 分类模型 | overripe | 过熟（记录） |

## 数据集

数据集来源于 [Roboflow](https://universe.roboflow.com/edge-computing-workspace/strawberry-stem-label-labelme/dataset/1)，采用 CC BY 4.0 许可协议，包含草莓和茎的标注。
