## Results

Our model achieves the following performance on :

### PASTIS - Panoptic segmentation

Our spatio-temporal encoder U-TAE combined with our PaPs instance segmentation module achieves 40.4 Panoptic Quality (PQ) on PASTIS for panoptic segmentation.
When replacing U-TAE with a convolutional LSTM the performance drops to 33.4 PQ.

| Model name         | SQ  | RQ | PQ|
| ------------------ |--- | --- |--- |
| **U-TAE** (ours)      | **81.5**|**53.2** |**43.8**|
| UConvLSTM+PaPs  | 80.2|   43.9   |  35.6|

### PASTIS - Semantic segmentation
Our spatio-temporal encoder U-TAE yields a semantic segmentation score of 63.1 mIoU on PASTIS, achieving an improvement of approximately 5 points compared to the best existing methods that we re-implemented (Unet-3d, Unet+ConvLSTM and Feature Pyramid+Unet).
See the paper for more details.

| Model name         | #Params| OA  |  mIoU |
| ------------------ |---- |---- | ---|
| **U-TAE**  (ours) |   **1.1M**|  **83.2%**    | **63.1%**|
| Unet-3d   | 1.6M|    81.3%    |  58.4%|
| Unet-ConvLSTM |1.5M  |     82.1%    |  57.8%|
| FPN-ConvLSTM  | 1.3M|    81.6%   |  57.1%|



## Requirements

### PASTIS Dataset download
The Dataset is freely available for download [here](https://github.com/VSainteuf/pastis-benchmark). 


### Python requirements
To install requirements:

```setup
pip install -r requirements.txt
```

(`torch_scatter` is required for the panoptic experiments. 
Installing this library requires a little more effort, see [the official repo](https://github.com/rusty1s/pytorch_scatter))


## Inference with pre-trained models

### Panoptic segmentation


Pre-trained weights of U-TAE+Paps are available [here]([weights file ](https://drive.google.com/drive/folders/1XQY96g6uoAGLpDUq4qfcsJ6zf4b5H0VN?usp=drive_link))

To perform inference of the pre-trained model on the test set of PASTIS run:

```test
python test_panoptic.py --dataset_folder PATH_TO_DATASET --weight_folder PATH_TO_WEIGHT_FOLDER --res_dir OUPUT_DIR
```


### Semantic segmentation


Pre-trained weights of U-TAE are available [here]([Weights file](https://drive.google.com/drive/folders/1CXeq_Sn7RRHUAOYEUvbzUpQkG5Ubb8NM?usp=drive_link))

To perform inference of the pre-trained model on the test set of PASTIS run:

```test
python test_semantic.py --dataset_folder PATH_TO_DATASET --weight_folder PATH_TO_WEIGHT_FOLDER --res_dir OUPUT_DIR
```


## Training models from scratch

### Panoptic segmentation

To reproduce the main result for panoptic segmentation (with U-TAE+PaPs) run the following :

```train
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR
```
Options are also provided in `train_panoptic.py` to reproduce the other results of Table 2:

```train
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_NoCNN --no_mask_conv
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_UConvLSTM --backbone uconvlstm
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_shape24 --shape_size 24
```

Note: By default this script runs the 5 folds of the cross validation, which can be quite long (~12 hours per fold on a Tesla V100). 
Use the fold argument to execute one of the 5 folds only 
(e.g. for the 3rd fold : `python train_panoptic.py --fold 3 --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR`).

### Semantic segmentation

To reproduce results for semantic segmentation (with U-TAE) run the following :

```train
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR
```

And in order to obtain the results of the competing methods presented in Table 1 :

```train
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_UNET3d --model unet3d
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_UConvLSTM --model uconvlstm
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_FPN --model fpn
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_BUConvLSTM --model buconvlstm
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_COnvGRU --model convgru
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_ConvLSTM --model convlstm

```
Finally, to reproduce the ablation study presented in Table 1 :

```train
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_MeanAttention --agg_mode att_mean
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_SkipMeanConv --agg_mode mean
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_BatchNorm --encoder_norm batch
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_SingleDate --mono_date "08-01-2019"

```

