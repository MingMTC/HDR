# README

## Backgroud 

This is a demo for HDR approach proposed by KDD 2023 Accepted Paper: Capturing Conversion Rate Fluctuation during Sales Promotions: A Novel Historical Data Reuse Approach [[paper]](https://arxiv.org/abs/2305.12837). 

This demo code is built on a simplified version of our production model because we must omit the details of the original model for anonymity. We provide it to illustrate the details of TransBlock and Distribution Shift Correction (DSC).

+ `main.py` : The main workflow. 
+ `model.py` : The CVR prediction model. 
+ `model_utils.py` : The modules. 


## Quickstart  

We test the code in Miniconda 4.13.0 (64-bit) | Python 3.6.6 | tensorflow 1.15.0. 

Before running the code, make sure you have setup the tensorflow 1.15.0. 

```bash
pip install tensorflow==1.15.0
```

Then, you could run our code by 

```bash
python main.py
```

## Citation 

We appreciate your citation if you think our work is helpful. 

```
@inproceedings{chan2023capturing,
  title={Capturing Conversion Rate Fluctuation during Sales Promotions: A Novel Historical Data Reuse Approach},
  author={Chan, Zhangming and Zhang, Yu and Han, Shuguang and Bai, Yong and Sheng, Xiang-Rong and Lou, Siyuan and Hu, Jiacen and Liu, Baolin and Jiang, Yuning and Xu, Jian and others},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={3774--3784},
  year={2023}
}
``` 
