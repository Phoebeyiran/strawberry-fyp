"""训练草莓成熟度分类模型（YOLOv8-cls）

数据集目录结构要求:
    strawberry_ripeness_dataset/
    ├── train/
    │   ├── ripe/
    │   ├── unripe/
    │   └── overripe/
    ├── val/
    │   ├── ripe/
    │   ├── unripe/
    │   └── overripe/
    └── test/
        ├── ripe/
        ├── unripe/
        └── overripe/

用法:
    python train_cls.py                                    # 使用默认参数
    python train_cls.py --data path/to/my_dataset          # 指定数据集路径
    python train_cls.py --epochs 30 --device cpu           # 自定义参数
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="训练草莓成熟度分类模型")
    parser.add_argument("--model",    default="yolov8n-cls.pt",              help="预训练模型路径 (default: yolov8n-cls.pt)")
    parser.add_argument("--data",     default="strawberry_ripeness_dataset", help="分类数据集目录 (default: strawberry_ripeness_dataset)")
    parser.add_argument("--epochs",   type=int, default=50,                  help="训练轮数 (default: 50)")
    parser.add_argument("--batch",    type=int, default=32,                  help="批大小 (default: 32)")
    parser.add_argument("--imgsz",    type=int, default=224,                 help="输入图像尺寸 (default: 224)")
    parser.add_argument("--device",   default="mps",                         help="训练设备: cpu / mps / 0 (default: mps)")
    parser.add_argument("--patience", type=int, default=15,                  help="早停耐心值 (default: 15)")
    parser.add_argument("--name",     default="strawberry_ripeness",         help="实验名称")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        name=args.name,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        cos_lr=False,
        close_mosaic=10,
        amp=True,
        seed=0,
        deterministic=True,
    )
    print(f"训练完成，结果保存在 runs/classify/{args.name}/")


if __name__ == "__main__":
    main()
