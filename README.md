# GPTrans [[Paper](https://arxiv.org/abs/2305.11424)]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-propagation-transformer-for-graph/graph-regression-on-pcqm4m-lsc)](https://paperswithcode.com/sota/graph-regression-on-pcqm4m-lsc?p=graph-propagation-transformer-for-graph)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-propagation-transformer-for-graph/node-classification-on-pattern)](https://paperswithcode.com/sota/node-classification-on-pattern?p=graph-propagation-transformer-for-graph)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-propagation-transformer-for-graph/node-classification-on-cluster)](https://paperswithcode.com/sota/node-classification-on-cluster?p=graph-propagation-transformer-for-graph)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-propagation-transformer-for-graph/graph-regression-on-pcqm4mv2-lsc)](https://paperswithcode.com/sota/graph-regression-on-pcqm4mv2-lsc?p=graph-propagation-transformer-for-graph)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-propagation-transformer-for-graph/graph-regression-on-zinc-500k)](https://paperswithcode.com/sota/graph-regression-on-zinc-500k?p=graph-propagation-transformer-for-graph)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-propagation-transformer-for-graph/graph-property-prediction-on-ogbg-molhiv)](https://paperswithcode.com/sota/graph-property-prediction-on-ogbg-molhiv?p=graph-propagation-transformer-for-graph)

<!-- ## Description -->

This paper presents a novel transformer architecture for graph representation learning. The core insight of our method is to fully consider the information propagation among nodes and edges in a graph when building the attention module in the transformer blocks. Specifically, we propose a new attention mechanism called Graph Propagation Attention (GPA). It explicitly passes the information among nodes and edges in three ways, i.e. node-to-node, node-to-edge, and edge-to-node, which is essential for learning graph-structured data. On this basis, we design an effective transformer architecture named Graph Propagation Transformer (GPTrans) to further help learn graph data. We verify the performance of GPTrans in a wide range of graph learning experiments on several benchmark datasets. These results show that our method outperforms many state-of-the-art transformer-based graph models with better performance.

<img width="511" alt="image" src="https://github.com/czczup/GPTrans/assets/23737120/32f064ab-0e37-4efd-8c30-e2c2c7d81f42">


## 🗓️ Schedule

- [ ] MolPCBA & TSP models
- [x] Release code and models

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

**PCQM4Mv2**

| Model     | #Param | validate MAE | test MAE                                                           | Config                                                 | Model                                                                                  |
|:---------:|:------:|:------------:|:------------------------------------------------------------------:|:------------------------------------------------------:|:--------------------------------------------------------------------------------------:|
| GPTrans-T | 6.6M   | 0.0833       | [0.0842](https://ogb.stanford.edu/docs/lsc/leaderboards/#pcqm4mv2) | [config](configs/pcqm4mv2/gptrans_tiny_pcqm4mv2.yaml)  | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_tiny_pcqm4mv2.pth)  |
| GPTrans-S | 13.6M  | 0.0823       | -                                                                  | [config](configs/pcqm4mv2/gptrans_small_pcqm4mv2.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_small_pcqm4mv2.pth) |
| GPTrans-B | 45.7M  | 0.0813       | -                                                                  | [config](configs/pcqm4mv2/gptrans_base_pcqm4mv2.yaml)  | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_base_pcqm4mv2.pth)  |
| GPTrans-L | 86.0M  | 0.0809       | [0.0821](https://ogb.stanford.edu/docs/lsc/leaderboards/#pcqm4mv2) | [config](configs/pcqm4mv2/gptrans_large_pcqm4mv2.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_large_pcqm4mv2.pth) |

**PCQM4M**

| Model     | #Param | validate MAE | Config                                             | Model                                                                                |
|:---------:|:------:|:------------:|:--------------------------------------------------:|:------------------------------------------------------------------------------------:|
| GPTrans-T | 6.6M   | 0.1179       | [config](configs/pcqm4m/gptrans_tiny_pcqm4m.yaml)  | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_tiny_pcqm4m.pth)  |
| GPTrans-S | 13.6M  | 0.1162       | [config](configs/pcqm4m/gptrans_small_pcqm4m.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_small_pcqm4m.pth) |
| GPTrans-B | 45.7M  | 0.1153       | [config](configs/pcqm4m/gptrans_base_pcqm4m.yaml)  | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_base_pcqm4m.pth)  |
| GPTrans-L | 86.0M  | 0.1151       | [config](configs/pcqm4m/gptrans_large_pcqm4m.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_large_pcqm4m.pth) |

**MolHIV**

| Model     | #Param | Test AP (paper) | Val AP | Test AP | Config                                            | Model                                                                               |
|:---------:|:------:|:---------------:|:------:|:-------:|:-------------------------------------------------:|:-----------------------------------------------------------------------------------:|
| GPTrans-B | 45.7M  | 81.26 ± 0.32    | 81.61  | 81.49   | [config](configs/molhiv/gptrans_base_molhiv.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_base_molhiv.pth) |

**MolPCBA**

| Model     | #Param | Test AP (paper) | Val AP | Test AP | Config                                              | Model                                                                                |
|:---------:|:------:|:---------------:|:------:|:-------:|:---------------------------------------------------:|:------------------------------------------------------------------------------------:|
| GPTrans-B | 45.7M  | 31.15 ± 0.16    | 31.66  | 31.49   | [config](configs/molpcba/gptrans_base_molpcba.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_base_molpcba.pth) |

**ZINC**

| Model        | #Param | Test MAE (paper) | Val MAE | Test MAE | Config                                        | Model                                                                             |
|:------------:|:------:|:----------------:|:-------:|:--------:|:---------------------------------------------:|:---------------------------------------------------------------------------------:|
| GPTrans-Nano | 554K   | 0.077 ± 0.009    | 0.1133  | 0.0770   | [config](configs/zinc/gptrans_nano_zinc.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_nano_zinc.pth) |

**PATTERN**

| Model        | #Param | Test Acc (paper) | Val Acc | Test Acc | Config                                              | Model                                                                                |
|:------------:|:------:|:----------------:|:-------:|:--------:|:---------------------------------------------------:|:------------------------------------------------------------------------------------:|
| GPTrans-Nano | 554K   | 86.731 ± 0.085   | 86.6229 | 86.7489  | [config](configs/pattern/gptrans_nano_pattern.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_nano_pattern.pth) |

**CLUSTER**

| Model        | #Param | Test Acc (paper) | Val Acc | Test Acc | Config                                              | Model                                                                                |
|:------------:|:------:|:----------------:|:-------:|:--------:|:---------------------------------------------------:|:------------------------------------------------------------------------------------:|
| GPTrans-Nano | 554K   | 78.069 ± 0.154   | 78.2748 | 78.0705  | [config](configs/cluster/gptrans_nano_cluster.yaml) | [model](https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_nano_cluster.pth) |

### Evaluation

> Note: Datasets will be downloaded automatically.

> Note: There is a bug in multi-GPU evaluation, I will fix it. Please use single GPU to evaluate now.

<details>
<summary> To evaluate GPTrans-T on PCQM4Mv2 with 1 GPU </summary>
<br>
<div>

```shell
wget https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_tiny_pcqm4mv2.pth
sh dist_train.sh configs/pcqm4mv2/gptrans_tiny_pcqm4mv2.yaml 1 --resume ./gptrans_tiny_pcqm4mv2.pth --eval
```

</div>
</details>

<details>
<summary> To evaluate GPTrans-S on PCQM4Mv2 with 1 GPU </summary>
<br>
<div>

```shell
wget https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_small_pcqm4mv2.pth
sh dist_train.sh configs/pcqm4mv2/gptrans_small_pcqm4mv2.yaml 1 --resume ./gptrans_small_pcqm4mv2.pth --eval
```

</div>
</details>

<details>
<summary> To evaluate GPTrans-B on PCQM4Mv2 with 1 GPU </summary>
<br>
<div>

```shell
wget https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_base_pcqm4mv2.pth
sh dist_train.sh configs/pcqm4mv2/gptrans_base_pcqm4mv2.yaml 1 --resume ./gptrans_base_pcqm4mv2.pth --eval
```

</div>
</details>

<details>
<summary> To evaluate GPTrans-L on PCQM4Mv2 with 1 GPU </summary>
<br>
<div>

```shell
wget https://huggingface.co/czczup/GPTrans/resolve/main/gptrans_large_pcqm4mv2.pth
sh dist_train.sh configs/pcqm4mv2/gptrans_large_pcqm4mv2.yaml 1 --resume ./gptrans_large_pcqm4mv2.pth --eval
```

</div>
</details>

### Training

<details>
<summary> To train GPTrans-T on PCQM4Mv2 with 8 GPU on 1 node </summary>
<br>
<div>

```shell
sh dist_train.sh configs/pcqm4mv2/gptrans_tiny_pcqm4mv2.yaml 8
```

</div>
</details>

<details>
<summary> To train GPTrans-S on PCQM4Mv2 with 8 GPU on 1 node </summary>
<br>
<div>

```shell
sh dist_train.sh configs/pcqm4mv2/gptrans_small_pcqm4mv2.yaml 8
```

</div>
</details>

<details>
<summary> To train GPTrans-B on PCQM4Mv2 with 8 GPU on 1 node </summary>
<br>
<div>

```shell
sh dist_train.sh configs/pcqm4mv2/gptrans_base_pcqm4mv2.yaml 8
```

</div>
</details>

<details>
<summary> To train GPTrans-L on PCQM4Mv2 with 8 GPU on 1 node </summary>
<br>
<div>

```shell
sh dist_train.sh configs/pcqm4mv2/gptrans_large_pcqm4mv2.yaml 8
```

</div>
</details>

## 🤝 Acknowledgement

Thanks to the open source of the following projects:

[Graphormer](https://github.com/microsoft/Graphormer) &#8194;
[EGT](https://github.com/shamim-hussain/egt_pytorch) &#8194;
[GraphGPS](https://github.com/rampasek/GraphGPS) &#8194;
[OGB](https://github.com/snap-stanford/ogb) &#8194;

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
