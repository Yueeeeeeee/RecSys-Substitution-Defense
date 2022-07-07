# Introduction

The RecSys-Substitution-Defense repository is the PyTorch Implementation of RecSys 2022 Paper [Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders](https://arxiv.org/abs/2109.01165)

<img src=pics/intro.png>

We propose a substitution-based adversarial attack algorithm, which modifies the input sequence by selecting certain vulnerable elements and substituting them with adversarial items. In both untargeted and targeted attack scenarios, we observe significant performance deterioration using the proposed profile pollution algorithm. Motivated by such observations, we design an efficient adversarial defense method called Dirichlet neighborhood sampling. Specifically, we sample item embeddings from a convex hull constructed by multi-hop neighbors to replace the original items in input sequences. During sampling, a Dirichlet distribution is used to approximate the probability distribution in the neighborhood such that the recommender learns to combat local perturbations. Additionally, we design an adversarial training method tailored for sequential recommender systems. In particular, we represent selected items with one-hot encodings and perform gradient ascend on the encodings to search for the worst case linear combination of item embeddings in training. As such, the embedding function learns robust representations and the trained recommender is resistant to test-time adversarial examples. Extensive experiments show the effectiveness of both our attack and defense methods, which consistently outperform baselines by a significant margin across model architectures and datasets. 


## Citing 

Please consider citing the following paper if you use our methods in your research:
```
@inproceedings{yue2021black,
  title={Black-Box Attacks on Sequential Recommenders via Data-Free Model Extraction},
  author={Yue, Zhenrui and He, Zhankui and Zeng, Huimin and McAuley, Julian},
  booktitle={Proceedings of the 15th ACM Conference on Recommender Systems},
  year={2021}
}
```
```
@inproceedings{yue2022defending,
  title={Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders},
  author={Yue, Zhenrui and Zeng, Huimin and Kou, Ziyi and Shang, Lanyu and Wang, Dong},
  booktitle={Proceedings of the 16th ACM Conference on Recommender Systems},
  year={2022}
}
```


## Requirements

PyTorch, pandas, wget, libarchive-c, tqdm. For our running environment see requirements.txt


## Train Clean Recommender Models

```bash
python train.py
```
Excecute the above command (with arguments) to train a clean sequential recommender, select datasets from Movielens 1M/20M, Beauty, Games, LastFM, Steam and Yoochoose. Availabel models are NARM, SASRec, Locker and BERT4Rec. Trained models could be found under ./experiments/model-code/dataset-code/models/best_acc_model.pth


## Train Recommender Models with Defense Methods

```bash
python train_robust.py --defense_method=dirichlet
```
```bash
python train_robust.py --defense_method=advtrain
```
Excecute the first command (with arguments) to train with Dirichlet neighborhood sampling, excecute the second command (with arguments) to perform adversarial training, select datasets from Movielens 1M/20M, Beauty, Games, LastFM, Steam and Yoochoose. Availabel models are NARM, SASRec, Locker and BERT4Rec. Trained models could be found under ./experiments/defense/model-code/dataset-code/models/best_acc_model.pth


## Attack Trained Recommender Models

```bash
python attack.py
```
Run the above command (with arguments) to perform profile pollution attacks, select datasets from Movielens 1M/20M, Beauty, Games, LastFM, Steam and Yoochoose. Availabel models are NARM, SASRec, Locker and BERT4Rec. Add defense_method argument to attack models trained with our defense methods


## Performance

Recommender systems are first trained and attacked with the proposed substitution-based attack (see first table below). We also train robust recommenders with the proposed defense methods and evaluate the performance variations (see second table below).

### Profile Pollution Performance

<img src=pics/attack.png width=1000>

### Defense Performance

<img src=pics/defense.png width=1000>


## Acknowledgement

During the implementation we base our code mostly on [Dirichlet Neighborhood Ensemble](https://github.com/dugu9sword/dne) from Yi Zhou and [BERT4Rec](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch) by Jaewon Chung. Many thanks to these authors for their great work!
