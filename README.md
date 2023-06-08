# Multi-center study on predicting breast cancer lymph node status from core needle biopsy specimens using multi-modal and multi-instance deep learning

## Install

### Dependencies

- Python >= 3.6.0
- torch >= 1.7.0
- pandas >= 1.1.5
- numpy >= 1.18.5
- scikit-learn >= 0.23.2
- scipy >= 1.5.2
- albumentations >= 0.4.6
- opencv-python >= 4.3.0.36
- rich >= 9.3.0
- pytorch-tabnet >= 2.0.1
- efficientnet-pytorch >= 0.7.0 

## Data

The data of this work could be requested by contacting with the corresponding author. The data can be used only for "non-commercial" purposes and under the permission of the corresponding author.

## Checkpoint

The model checkpoint developed based on this clinical cohort could be found in checkpoint.7z file. 

## Usage

The test multimodal data (Whole Slide Imaging (WSI) and tabular data) should be pre-processed by first extracting offline patch features and then merging the patch features for each WSI.

The demo data in model input format could be found in sampledata.7z file.
 
- Extract offline features of each patch
```
python3 preprocessing/extract_feat_with_tta.py --level= x5
python3 preprocessing/extract_feat_with_tta.py --level= x10
python3 preprocessing/extract_feat_with_tta.py --level= x20
```

- Merge patch features for each WSI

```bash
python3 preprocessing/merge_patch_feat.py
```

- Model Inference based on multi-modal data

```bash
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port=XXXX model_inference.py --cfg ./configs/test.yaml
```

- Expected output

The expected output of model inference is the prediction of the lymph node metastasis of each individual patient (no metastasis, ITCs, micrometastasis, and macrometastasis).

## Disclaimer
This tool is for research purpose and not approved for clinical use.

This is not an official Tencent product.

## Coypright

This tool is developed in Tencent AI Lab.

The copyright holder for this project is Tencent AI Lab.

All rights reserved.