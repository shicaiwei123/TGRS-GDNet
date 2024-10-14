# MSHNet
Code for Gradient Deocouped Learning with Unimodal Regularization for Multimodal Remote Sensing Image Classification

## Dependency
- Ubuntu20.04
- CUDA11.3
- PyToch1.12
- python3.8

## Dataset
- Download
  - Original [Huston20013](https://github.com/danfenghong/ISPRS_S2FL), [Augsburg](https://github.com/danfenghong/ISPRS_S2FL)
  - Preprocessing [Huston2013](https://drive.google.com/drive/folders/1YSbAFzD9MKcNMBbYTeax_c1XNkSjZC_a) [Augsburg](https://drive.google.com/drive/folders/1f4bvCefoJ9Xd6QTbByDSBY5x7pAW1u2q), [berlin](https://pan.baidu.com/s/10Cx9Rpqu03n5WsUS2x8tpw?pwd=f2fh)
  - precessing code: https://github.com/danfenghong/IEEE_TGRS_GCN/tree/master/DataGeneration_Functions


- Dataset convert
  - since the processed dataset in berlin and augsburg dataset, such as hsi.dat, is very large, it's hard to load them in a small GPU. We decouple the single  '.dat' file to multiple ‘.npy’ file  with [dataset_convert.py](test/dataset_convert.py)
  - we also provide the converted dataset [Augsburg](https://pan.baidu.com/s/1KwUOHDXxIRwX3OArZ_2vPg?pwd=q7hc), [Berlin](https://pan.baidu.com/s/1vjF0iNoNyLUhgNE_bC_icw?pwd=vayx)


- Build soft link
  ```bash
  cd GDNet
  mkdir data
  ln -s path_to_download_data ./data/dataset_name
  
  for example: ln -s /home/data/shicaiwei/remote_sensing/huston2013 ./data/huston2013
  ```


## Train

### Name Rules of Bash Files 
- dataset_operation_modality_.sh

  - dataset
    - huston2013
    - Augsburg
  - operation
    - F: Fusing multimodal data
  - modality
    - H: HSI modality
    - S: Sar modality
    - L: LiDAR modality
    - M: MS modality
    - D: DSM modality

### Train process
  - To average the results, for each sub-task, we train three models and choose the one with middle performance for the following task. 




### Train multimodal  model with GDL
```bash
cd src
bash huston2013_F_HL_GDL.sh
bash berlin_F_HS_GDL.sh
bash augsburg_F_HS_GDL.sh
bash augsburg_F_HD_GDL.sh
```



### Train multimodal  baseline
```bash
cd src
bash huston2013_F_HL_X.sh
bash berlin_F_HS_X.sh
bash augsburg_F_HS_X.sh
bash augsburg_F_HD_X.sh
```



### Test
```bash
cd test
python multimodal_baseline_test.py 0 0 0 0 
```




## Visualization
- Prepairation
  - the patch of each pixel from the dataset. You can get those with the precessing code: https://github.com/danfenghong/IEEE_TGRS_GCN/tree/master/DataGeneration_Functions by processed all image pixels. 
  - the pretrained model
  - details can be seen in the function of **huston_prediction_plot** and **augsburg_prediction_plot** in prediction_plot.py  
- Code
    ```bash
    cd test
    python prediction_plot.py
    ```