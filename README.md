# Sentiment Analysis with DistilBERT

## 📌 项目简介
本项目使用 **Hugging Face Transformers** 微调 **DistilBERT** 模型，在 IMDb 数据集上进行 **情感分析（二分类）**。

## 📂 项目结构
项目目录与核心文件的功能说明如下：
```bash
SentimentClassification/
├── data/               # 数据存储目录（含原始数据与预处理后子集）
│   ├── train.csv       # 预处理后的训练子集
│   └── val.csv         # 预处理后的验证子集
├── results/            # 训练结果目录
├── logs/               # 训练日志目录
├── plots/              # 可视化结果目录
├── best_epoch_model/   # 最佳模型目录
├── data_loader.py      # 数据处理模块：加载原始数据、生成并保存训练/验证子集
├── model_loader.py     # 模型加载模块：初始化 DistilBERT 预训练模型与分词器
├── plot_utils.py       # 可视化工具模块：绘制训练过程的损失、准确率变化曲线
├── utils.py            # 辅助工具模块：包含文本预处理函数、评估指标计算函数
├── train.py            # 训练主程序：整合数据、模型、训练逻辑，输出最佳模型与可视化结果
├── requirements.txt    # Python 依赖库清单
└── README.md           # 项目说明文档
```

## ⚙️ 环境设置
```bash
pip install -r requirements.txt
```

## 🚀 运行步骤
```bash
python train.py
```

## 📊 输出结果
- 训练日志保存在 `./logs/`
- 训练结果和模型保存在 `./results/`
- 最佳模型保存在 `./best_epoch_model/`
- 可视化结果保存在`./plots/`
- 评估指标包括 Accuracy, F1-score, precision 和 recall.

