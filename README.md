# CIFAR‑10 图像分类课程项目


* 一键生成数据集示例图（≥ 6）、训练曲线（4 张）、混淆矩阵、正确/错误预测示例 
* 轻量级 CNN，30 轮即可在 CPU 上取得 ~70%+ 测试准确率  
* 代码行数 ≈ 200，适合“核心代码分析”截图讲解

## 快速开始
```bash
pip install torch torchvision matplotlib scikit-learn numpy
# 训练
python train.py --data_dir <cifar-10-batches-py> --out_dir outputs
# 数据集示例图
python visualize_dataset.py --data_dir <cifar-10-batches-py> --out_dir outputs
# 生成混淆矩阵 & 示例预测
python evaluate.py --data_dir <cifar-10-batches-py> --ckpt outputs/best_model.pth --out_dir outputs
```

生成文件一览：
```
outputs/
├── best_model.pth
├── confusion_matrix.png
├── correct_samples.png
├── wrong_samples.png
├── train_loss.png
├── test_loss.png
├── train_acc.png
├── test_acc.png
├── train_samples.png
└── test_samples.png
```
