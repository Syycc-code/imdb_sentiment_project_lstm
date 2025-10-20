# imdb_sentiment_project_lstm
ds_final_project
# IMDb 影评情感分析（LSTM 版本）

## 简介
本项目基于 Kaggle 上的 IMDb 影评数据集，使用 LSTM 进行情感分类，并通过 Streamlit 提供交互式界面：
- EDA（影评长度分布、词云、情感比例）
- 可训练 LSTM 模型（支持子集训练用于演示）
- 单条文本预测

## 数据集
- 来源（Kaggle）：IMDb Dataset of 50K Movie Reviews  
  链接: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- 文件名：`IMDB Dataset.csv`，请将其放在项目根目录

## 快速开始
1. 建议创建并激活 Python 虚拟环境  
2. 安装依赖：
```bash
pip install -r requirements.txt

