# GPTrans [[Paper](https://arxiv.org/abs/2305.11424)]

<!-- ## Description -->

This paper presents a novel transformer architecture for graph representation learning. The core insight of our method is to fully consider the information propagation among nodes and edges in a graph when building the attention module in the transformer blocks. Specifically, we propose a new attention mechanism called Graph Propagation Attention (GPA). It explicitly passes the information among nodes and edges in three ways, i.e. node-to-node, node-to-edge, and edge-to-node, which is essential for learning graph-structured data. On this basis, we design an effective transformer architecture named Graph Propagation Transformer (GPTrans) to further help learn graph data. We verify the performance of GPTrans in a wide range of graph learning experiments on several benchmark datasets. These results show that our method outperforms many state-of-the-art transformer-based graph models with better performance.

## 🗓️ Schedule

- [ ] Release code and models (We plan to release code and models in this month.)

## 🏠 Overview

<img width="826" alt="image" src="https://github.com/czczup/GPTrans/assets/23737120/105c3aae-ac65-4fac-9e4a-94ff5436508a">

## 🛠️ Installation

- Clone this repo:

```
git clone https://github.com/czczup/GPTrans
cd GPTrans
```

- Create a conda virtual environment and activate it:

```
conda create -n gptrans python=3.8 -y
conda activate gptrans
```

- Install torch==1.12 with CUDA==11.3:

```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

- Install other requirements:

```bash
pip install timm==0.6.12
pip install yacs==0.1.8
pip install dgl==1.0.1
pip install torch-geometric==1.7.2
pip install torch-scatter==2.0.9
pip install torch-sparse==0.6.14 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install ogb==1.3.2
pip install rdkit
pip install cython
pip install termcolor
pip install ninja
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## 🎯 Get Started

### Model Zoo

PCQM4Mv2

| Model     | #Param | validate MAE | Config                                                 | Model                                                                                  |
| --------- | ------ | ------------ | ------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| GPTrans-T | 6.6M   | 0.0833       | [config](configs/pcqm4mv2/gptrans_tiny_pcqm4mv2.yaml)  | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_tiny_pcqm4mv2.pth)  |
| GPTrans-S | 13.6M  | 0.0823       | [config](configs/pcqm4mv2/gptrans_small_pcqm4mv2.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_small_pcqm4mv2.pth) |
| GPTrans-B | 45.7M  | 0.0813       | [config](configs/pcqm4mv2/gptrans_base_pcqm4mv2.yaml)  | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_base_pcqm4mv2.pth)  |
| GPTrans-L | 86.0M  | 0.0809       | [config](configs/pcqm4mv2/gptrans_large_pcqm4mv2.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_large_pcqm4mv2.pth) |

PCQM4M

| Model     | #Param | validate MAE | Config                                             | Model                                                                                |
| --------- | ------ | ------------ | -------------------------------------------------- | ------------------------------------------------------------------------------------ |
| GPTrans-T | 6.6M   | 0.1179       | [config](configs/pcqm4m/gptrans_tiny_pcqm4m.yaml)  | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_tiny_pcqm4m.pth)  |
| GPTrans-S | 13.6M  | 0.1162       | [config](configs/pcqm4m/gptrans_small_pcqm4m.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_small_pcqm4m.pth) |
| GPTrans-B | 45.7M  | 0.1153       | [config](configs/pcqm4m/gptrans_base_pcqm4m.yaml)  | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_base_pcqm4m.pth)  |
| GPTrans-L | 86.0M  | 0.1151       | [config](configs/pcqm4m/gptrans_large_pcqm4m.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_large_pcqm4m.pth) |

MolPCBA

| Model     | #Param | Test AP (paper) | Test AP | Config     | Model     |
| --------- | ------ | --------------- | ------- | ---------- | --------- |
| GPTrans-B | 45.7M  | 31.15±0.16      |         | [config]() | [model]() |
| GPTrans-L | 86.0M  | 32.43±0.22      |         | [config]() | [model]() |

MolHIV

| Model     | #Param | Test AP (paper) | Test AP | Config     | Model     |
| --------- | ------ | --------------- | ------- | ---------- | --------- |
| GPTrans-B | 45.7M  | 81.26±0.32      |         | [config]() | [model]() |

### Evaluation

To evaluate an `GPTrans-T` on PCQM4Mv2 with 1 GPU, run:

```shell
wget https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_tiny_pcqm4mv2.pth
sh dist_train.sh configs/pcqm4mv2/gptrans_tiny_pcqm4mv2.yaml 1 --resume ./gptrans_tiny_pcqm4mv2.pth  --eval
```

To evaluate an `GPTrans-S` on PCQM4Mv2 with 1 GPU, run:

```shell
wget https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_small_pcqm4mv2.pth
sh dist_train.sh configs/pcqm4mv2/gptrans_small_pcqm4mv2.yaml 1 --resume ./gptrans_small_pcqm4mv2.pth  --eval
```

To evaluate an `GPTrans-B` on PCQM4Mv2 with 1 GPU, run:

```shell
wget https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_base_pcqm4mv2.pth
sh dist_train.sh configs/pcqm4mv2/gptrans_base_pcqm4mv2.yaml 1 --resume ./gptrans_base_pcqm4mv2.pth  --eval
```

To evaluate an `GPTrans-L` on PCQM4Mv2 with 1 GPU, run:

```shell
wget https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_large_pcqm4mv2.pth
sh dist_train.sh configs/pcqm4mv2/gptrans_large_pcqm4mv2.yaml 1 --resume ./gptrans_large_pcqm4mv2.pth  --eval
```

### Training

To train an `GPTrans-T` on PCQM4Mv2 with 8 GPU on 1 node, run:

```shell
sh dist_train.sh configs/pcqm4mv2/gptrans_tiny_pcqm4mv2.yaml 8
```

To train an `GPTrans-S` on PCQM4Mv2 with 8 GPU on 1 node, run:

```shell
sh dist_train.sh configs/pcqm4mv2/gptrans_small_pcqm4mv2.yaml 8
```

To train an `GPTrans-B` on PCQM4Mv2 with 8 GPU on 1 node, run:

```shell
sh dist_train.sh configs/pcqm4mv2/gptrans_base_pcqm4mv2.yaml 8
```

To train an `GPTrans-L` on PCQM4Mv2 with 8 GPU on 1 node, run:

```shell
sh dist_train.sh configs/pcqm4mv2/gptrans_large_pcqm4mv2.yaml 8
```

## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🖊️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{chen2023gptrans,
  title={Graph Propagation Transformer for Graph Representation Learning},
  author={Chen, Zhe and Tan, Hao and Wang, Tao and Shen, Tianrun and Lu, Tong and Peng, Qiuying and Cheng, Cheng and Qi, Yue},
  journal={arXiv preprint arXiv:2305.11424},
  year={2023}
}
```
