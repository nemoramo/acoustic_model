[TOC]

# RamoSpeech　开源音频模型

=================================================

## 简介

RamoSpeech是一款由[ramosmy](https://github.com/ramosmy)开源的Automated Speech Recognition框架，本仓库中存放的是开源框架中的音频模型，该音频模型使用[Pytorch](https://github.com/pytorch/pytorch)编写，基于较早的模型DeepCNN + DeepLSTM + FC + CTC实现．

## 模型

1. DFCNN
2. Come soon...

Come soon!

## 依赖库

| **[Horovod](https://github.com/horovod/horovod)** | **通用流行的分布式深度学习框架，详细了解可参见Horovod github** |
| :-----------------------------------------------: | :----------------------------------------------------------: |
| **[Pytorch](https://github.com/pytorch/pytorch)** |                 **本仓库采用的深度学习框架**                 |

以上所提是构建模型的最基本库，要保证该模型可以在你的机子上运行，还需要：

1. tesorboardX
2. tqdm
3. scipy, numpy

## 运行训练代码

1. DFCNN

   如果不加载预训练的模型的话：

   ```bash
   horovodrun -np YOUR_WORKER_NUMBERS -H localhost:YOUR_WORKER_NUMBERS python train.py \
           --data_type YOUR_DATA_TYPE --model_path YOUR_MODEL_PATH --model_name YOUR_MODEL_NAME \
           --gpu_rank YOUR_WORKER_NUMBERS --epochs 1000 --save_step 20 --batch_size YOUR_BATCH_SIZE
   ```

   加载预训练模型请添加　--load_model

   YOUR_NUM_WORKERS指线程数

   YOUR_DATA_TYPE指数据类型，分为all, thchs, aishell(陆续会增加primewords, st-cmds)

   请根据你的GPU数量来决定你的GPU_RANK