# www2023tiger
Codes for WWW 2023 paper "TIGER: Temporal Interaction Graph Embedding with Restarts" [arXiv](https://arxiv.org/abs/2302.06057)

# Data

Please download data from the [project homepage of JODIE](https://snap.stanford.edu/jodie/) and pre-process them with the script provided by [TGN](https://github.com/twitter-research/tgn).

# How to use

Temporal Link Prediction

```
python train_self_supervised.py --data [DATA] --msg_src [left/right] --upd_src [left/right] --restarter [seq/static] --restart_prob [0/0.001/...]
```
If you want to use mooc/lastfm datasets, please pass one more argument: `--dim 100`.

Temporal Link Prediction with multi-GPU
```
python train_self_supervised_ddp.py --gpu 0,1,2,3 [...and other arguments]
```

Node Classification
```
python train_supervised.py --code [CODE]
```
Here, [CODE] is the HASH code of a trained model with `train_self_supervised.py`.

# Cite

```
@inproceedings{zhang2023tiger,
  title={TIGER: Temporal Interaction Graph Embedding with Restarts},
  author={Zhang, Yao and Xiong, Yun and Liao, Yongxiang and Sun, Yiheng and Jin, Yucheng and Zheng, Xuehao and Zhu, Yangyong},
  booktitle={ACM Web Conference},
  year={2023}
}
```
