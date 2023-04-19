# AICITY2023_Track5
This repo includes solution for AICity2023 Challenge Track 5 - Detecting Violation of Helmet Rule for Motorcyclists

![framework](GeneralPipline.png)
# Installation
Python 3.7 or later with dependencies listed in requirements.txt. Run the below command for install dependencies:
```commandline
pip install -r requirements.txt
```
# Data Preparation For Training
Download the training data, which is provided by 2023 ACity Challenge Track 5 and put the download file into ./aicity_dataset/

In the video dataset, which we've broken down into frames that can be downloaded here 
.........


# Reproduce the result on AICity 2023 Challenge Track 5
Our solution implemented in 3 steps
## STEP 1 - Baseline training
### Train

Download the training data label zip file and extract it inside ./baseline_training/datasets/

Please correct the data path in the file config ./baseline_training/.....
```commandline
cd baseline_training
python effdet_train.py
.....
```
### Inference
```commandline
python predict.py
```
After running the executable, the result file will be received as a result_motorcyclist.txt in the ./baseline_training

## STEP 2 - Head training
Download the training data zip file and extract it inside ./head_training/datasets/

Please correct the data path in the file config ./head_training/.....
```commandline
cd head_training
python effdet_train.py
.....
```
### Inference
```commandline
python predict.py
```
After running the executable, the result file will be received as a result_head.txt in the ./head_training

## STEP 3 - Post processing for tracking
In this step we used the result files of step 1 and step 2 and combined with the module object association to get the final result

Please correct the resulting file path in infer.py

```commandline
python infer.py
```

## Note: We execute the training with one DGX node with 8 NVIDIA A100-40GB GPU


## Public Leaderboard
| TeamName           | mAP    |
|--------------------|--------|
| **IC_SmartVision** | 0.6997 |



## Citation

If you find our work useful, please cite the following:

```text
@inreview{Tran2022,  
    author={Viet Hung Duong and Quang Huy Tran and Huu Si Phuc Nguyen and Duc Quyen Nguyen and Tien Cuong Nguyen},  
    title={Helmet Rule Violation Detection for Motorcyclists using a Custom
Tracking Framework and Advanced Object Detection Techniques},  
    booktitle={CVPR Workshop},
    year={2023}  
}
```

## Contact
Viet Hung Duong (hungdv@vnpt.vn)

Quang Huy Tran (huytq@vnpt.vn)

Huu Si Phuc Nguyen (phucnhs@vnpt.vn)

Duc Quyen Nguyen (quyennd@vnpt.vn)

Tien Cuong Nguyen (cuongnt@vnpt.vn)