"""训练草莓+茎检测模型（YOLOv8）

用法:
    python train_det.py                    # 使用默认参数
    python train_det.py --epochs 50        # 自定义 epoch 数
    python train_det.py --device cpu       # 使用 CPU 训练
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="训练草莓+茎检测模型")
    parser.add_argument("--model",    default="yolov8n.pt",        help="预训练模型路径 (default: yolov8n.pt)")
    parser.add_argument("--data",     default="dataset/data.yaml", help="数据集配置文件 (default: dataset/data.yaml)")
    parser.add_argument("--epochs",   type=int, default=100,       help="训练轮数 (default: 100)")
    parser.add_argument("--batch",    type=int, default=16,        help="批大小 (default: 16)")
    parser.add_argument("--imgsz",    type=int, default=640,       help="输入图像尺寸 (default: 640)")
    parser.add_argument("--device",   default="mps",               help="训练设备: cpu / mps / 0 (default: mps)")
    parser.add_argument("--patience", type=int, default=20,        help="早停耐心值 (default: 20)")
    parser.add_argument("--name",     default="strawberry_stem_detection", help="实验名称")
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
    print(f"训练完成，结果保存在 runs/detect/{args.name}/")


if __name__ == "__main__":
    main()
