# DM2C
This is a PyTorch implementation of the DM2C model as described in our paper:

>Y. Jiang, Z. Yang, Q. Xu, X. Cao and Q. Huang. DM2C: Deep Mixed-Modal Clustering. NeurIPS 2019.

## Dependencies
- PyTorch >= 1.0
- numpy
- sklearn
- scipy

## Data
The processed mixed-modal datasets are provided in `data/`. You may find the original Wikipedia and NUS-WIDE-10K datasets (with fully paired samples) [here](https://github.com/sunpeng981712364/ACMR_demo/tree/master/data).

## Pre-train the modality-specific auto-encoders
Sample code:
```
python train.py --dataset wikipedia --data_dir data/wikipedia \
--pretrain img --lr_ae 1e-4 --log_freq 15 --n_epochs 200
python train.py --dataset wikipedia --data_dir data/wikipedia \
--pretrain txt --lr_ae 1e-5 --log_freq 15 --n_epochs 200
```

## Train the DM2C model
Sample code:
```
python3 train.py --dataset wikipedia --data_dir data/wikipedia \
--pretrain load_ae --img_cptpath cpt/wikipedia_img_pretrain_checkpt.pkl --txt_cptpath cpt/wikipedia_txt_pretrain_checkpt.pkl \
--lr_g 1e-4 --lr_d 5e-5 --lr_ae 5e-4 --gan_type wasserstein --n_epochs 220 --weight_decay 0 --lamda3 0.5 --lamda1 1 --cpt_dir cpt --seed 2018
```

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
