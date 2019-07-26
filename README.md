[TOC]

# RamoSpeech　开源音频模型

## 简介

RamoSpeech是一款由[ramosmy](https://github.com/ramosmy)开源的Automated Speech Recognition框架，本仓库中存放的是开源框架中的音频模型，该音频模型使用[Pytorch](https://github.com/pytorch/pytorch)编写，基于较早的模型DeepCNN + DeepLSTM + FC + CTC实现．

## 模型

1. DFCNN

   ```bash
   AcousticModel(
     (dropout): Dropout(p=0.5, inplace=False)
     (conv1): Sequential(
       (conv1_conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
       (conv1_norm1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       (conv1_relu1): ReLU()
       (conv1_dropout1): Dropout(p=0.1, inplace=False)
       (conv1_conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (conv1_norm2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       (conv1_relu2): ReLU()
       (conv1_maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
       (conv1_dropout2): Dropout(p=0.1, inplace=False)
     )
     (conv2): Sequential(
       (conv2_conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (conv2_norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       (conv2_relu1): ReLU()
       (conv2_dropout1): Dropout(p=0.1, inplace=False)
       (conv2_conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (conv2_norm2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       (conv2_relu2): ReLU()
       (conv2_maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
       (conv2_dropout2): Dropout(p=0.1, inplace=False)
     )
     (conv3): Sequential(
       (conv3_conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (conv3_relu1): ReLU()
       (conv3_dropout1): Dropout(p=0.2, inplace=False)
       (conv3_norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       (conv3_conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (conv3_norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       (conv3_relu2): ReLU()
       (conv3_maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
       (conv3_dropout2): Dropout(p=0.2, inplace=False)
     )
     (conv4): Sequential(
       (conv4_conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (conv4_norm1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       (conv4_relu1): ReLU()
       (conv4_dropout1): Dropout(p=0.2, inplace=False)
       (conv4_conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (conv4_relu2): ReLU()
       (conv4_conv3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (conv4_norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       (conv4_relu3): ReLU()
       (conv4_dropout2): Dropout(p=0.2, inplace=False)
     )
     (fc1): Linear(in_features=3200, out_features=128, bias=True)
     (fc2): Linear(in_features=256, out_features=128, bias=True)
     (fc3): Linear(in_features=128, out_features=1215, bias=True)
     (rnn): LSTM(128, 128, num_layers=4, batch_first=True, dropout=0.1, bidirectional=True)
   )
   ['wang3', 'luo4', 'shang4', 'yi1', 'zhang1', 'yong3', 'jia1', 'qiao2', 'tou2', 'mo3', 'ji4', 'fan4', 'dian4', 'de', 'jie2', 'zhang4', 'dan1', 'shi2', 'fen1', 'yin3', 'ren2', 'zhu4', 'mu4']
   ['wang3', 'luo4', 'shang4', 'yi1', 'zhang1', 'yong3', 'jia1', 'qiao2', 'tou2', 'guo2', 'ji4', 'fan4', 'dian4', 'de', 'jie2', 'zhang4', 'dan1', 'shi2', 'fen1', 'yin3', 'ren2', 'zhu4', 'mu4']
   Prediction using 1.78259s
   ```

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

安装上述所需文件，只需要：

```bash
pip install -r requirments.txt
```

推荐使用virtuenv新开一个环境，当然也可以使用

```bash
conda create -n YOUR_NEW_ENV python=3.7
```

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

## 数据获取

Come soon!

## 解码器

1. 本仓库附有一个建议的BeamSearch解码器，参考BeamSearch.py文件．但是运行速度较慢，解码需用时25秒左右(BeamWidth=10)，不建议采用．
2. 本仓库另提供一个由Baidu DeepSpeech2提供的解码器，参考ctcDecode.py文件，当然百度提供的是character-level的，本仓库对该文件做了部分改动，使其成为word-level的，读者可以自行参考对比．

## 引用他人

1. [ASRT_SpeechRecognition](https://github.com/nl8590687/ASRT_SpeechRecognition)
2. [DeepSpeech2,Baidu](https://github.com/PaddlePaddle/DeepSpeech)