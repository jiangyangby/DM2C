# DM2C
This is a PyTorch implementation of the DM2C model as described in our paper:

>Y. Jiang, Z. Yang, Q. Xu, X. Cao and Q. Huang. DM2C: Deep Mixed-Modal Clustering. NeurIPS 2019.

## Dependencies
- PyTorch >= 1.0
- numpy
- sklearn
- scipy
- munkres

## Data
The preprocessed mixed-modal datasets are provided in `data/`. You may find the original Wikipedia and NUS-WIDE-10K datasets (with fully paired samples) [here](https://github.com/sunpeng981712364/ACMR_demo/tree/master/data).

Note that the preprocessing is a PCA on the original extracted features for each modality to reduce them into the same dimension (so that we can evaluate traditional methods like k-means). Thus the provided data include 2048d features for Wikipedia and 1000d features for NUS-WIDE-10K.

## Pre-train the modality-specific auto-encoders
Sample command:
```
python train.py --dataset wikipedia --data_dir data/wikipedia \
--pretrain img --lr_ae 1e-4 --log_freq 15 --n_epochs 150
python train.py --dataset wikipedia --data_dir data/wikipedia \
--pretrain txt --lr_ae 1e-5 --log_freq 15 --n_epochs 150
```

## Train the DM2C model
* Set the learning rate for cross-modal mappings (generators) `lr_g`, discriminators `lr_d`, and auto-encoders `lr_ae`.
* Set the coefficient for cycle consistency loss `lamda1` and adversarial loss `lamda3`.

Here is a sample command:
```
python3 train.py --dataset wikipedia --data_dir data/wikipedia \
--pretrain load_ae --img_cptpath cpt/wikipedia_img_pretrain_checkpt.pkl --txt_cptpath cpt/wikipedia_txt_pretrain_checkpt.pkl \
--lr_g 1e-4 --lr_d 5e-5 --lr_ae 5e-4 --gan_type wasserstein --n_epochs 220 --weight_decay 0 --lamda3 0.5 --lamda1 1 --cpt_dir cpt --seed 2018
```
(I'm not sure if this is the final version of the code. But I will check the code again when I have spare time.)

## Citation
Please cite our paper if you use this code in your own work:

```
@inproceedings{jiang2019dm2c,
  title={DM2C: Deep Mixed-Modal Clustering},
  author={Jiang, Yangbangyan and Xu, Qianqian and Yang, Zhiyong and Cao, Xiaochun and Huang, Qingming},
  booktitle={Advances in Neural Information Processing Systems},
  pages={5880--5890},
  year={2019}
}
```
