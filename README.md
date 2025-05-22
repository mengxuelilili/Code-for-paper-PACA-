# 项目简介

本项目包含数据集处理、模型训练和模型架构的代码。以下是各个文件夹和文件的简要说明：

## 文件夹结构

- `data/`: 存放原始数据。
- `dataset/`: 包含数据预处理和加载的代码。
  - `datadeal.py`: 数据处理脚本，负责数据的预处理和嵌入。
- `model/`: 包含模型训练相关的代码。
  - `train.py`: 模型训练脚本。
- `models/`: 包含模型架构的代码。
  - `roformerccnn.py`: 模型架构介绍。

## Python脚本

- `pltantigen.py`: 用于抗原处理的脚本。
- `pltattentioncdr.py`: 用于CDR注意力处理的脚本。
- `pltCDR.py`: 用于CDR处理的脚本。
- `predict.py`: 用于模型预测的脚本。

## 使用说明

1. **数据预处理**:
   - 运行 `dataset/datadeal.py` 进行数据预处理。

2. **模型训练**:
   - 运行 `model/train.py` 进行模型训练。

3. **模型预测**:
   - 运行 `predict.py` 使用训练好的模型进行预测。

4. **查看模型架构**:
   - 查看 `models/roformercnn.py` 了解模型架构。

## 依赖

请确保安装了以下Python库：
- numpy
- pandas
- matplotlib
- torch
- torchvision

