# [CVPR 2022] Sequential Voting with Relational Box Fields for Active Object Detection

The official code release for our paper
```
Sequential Voting with Relational Box Fields for Active Object Detection
Qichen Fu, Xingyu Liu, Kris M. Kitani
CVPR2022
```

[[Paper](https://arxiv.org/abs/2110.11524)][[Project](https://fuqichen1998.github.io/SequentialVotingDet/)][[Code](https://github.com/fuqichen1998/SequentialVotingDet)]

## Setup
1. Clone this repository
    ```
    git clone https://github.com/fuqichen1998/SequentialVotingDet
    ```
2. Create a python environment and install the requirements
    ```
    conda create --name seqvotingdet python=3.8
    conda activate seqvotingdet
    pip install -r requirements.txt
    ```

## Experiments

### Dataset Preparation
#### 100DOH
Download and unzip the [raw.zip](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/agreement.html?raw) and [file.zip](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/downloads/file.zip) following the official [100DOH](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/download.html) website, and put them under `100DOH_FOLDER`. Modify the data folder in [doh100.py](data_loader/doh100.py) accordingly.

#### MECCANO
Download and unzip the [Active Objects Bounding Box Annotations and Frames](https://iplab.dmi.unict.it/MECCANO/downloads/MECCANO_active_objects_annotations_frames.zip) and put it under `MECCANO_FOLDER`. Then download the pseudo labels in the [Google Drive](https://drive.google.com/drive/folders/1KHSOtmrhwtALnzTDlrutPzRIw_lgh10m?usp=sharing) and put them under `MECCANO_FOLDER/home/fragusa/`. Modify the data folder in [meccano.py](data_loader/meccano.py) accordingly.


### Evaluation
Please download the pre-trained checkpoints of our model in the [Google Drive](https://drive.google.com/drive/folders/1GS4zepeUngzrOhLlvkHbpg7qSaHcdeft?usp=sharing), and put them under [saved/models/](saved/models/). Then donwload the pre-computed annotation files in the [Google Drive](https://drive.google.com/drive/folders/1I8uhQFWNq2wCLFe7-Ac0JsN4bqCLSRWb?usp=sharing) and put them under [saved/](saved/cache/).

#### 100DOH
To evaluate on 100DOH, run the following command:
```
python test.py --resume saved/models/exp_doh100/checkpoint-epoch5.pth  -d 0 --ngpu 1 --use_gt_hand ""
python evaluate_100doh.py
```

#### MECCANO
To evaluate on MECCANO, run the following command:
```
python test.py --resume saved/models/exp_meccano/checkpoint-epoch5.pth -d 0 --ngpu 1
python evaluate_meccano.py
```

### Train
Download the detected hand bounding boxes in the [Google Drive](https://drive.google.com/file/d/1U4HrGmMRMTrUZkQep1dsKbMD5rvKNxiN/view?usp=sharing) and put it under [saved/](saved/).

#### 100DOH
To train on 100DOH, run the following command first to pretrain:
```
python train.py -c configs/doh100_dlv3+tr.json -d "0, 1, 2, 3"
```

Then run the following command to finetune the model using RL:
```
python train.py -c configs/doh100_dlv3+tr_rl.json -d "0, 1, 2, 3"
```

#### MECCANO
To train on MECCANO, run the following command first to pretrain:

```
python train.py -c configs/mcn_dlv3+tr.json -d "0, 1, 2, 3"
```

Then run the following command to finetune the model using RL:
```
python train.py -c configs/mcn_dlv3+tr_rl.json -d "0, 1, 2, 3"
```

Finally, please follow the Evaluation section to test the trained model.


## Citation
Please consider citing our paper if it is helpful:
```
@inproceedings{fu2021sequential,
  title={Sequential Decision-Making for Active Object Detection from Hand},
  author={Fu, Qichen and Liu, Xingyu and Kitani, Kris M},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Acknowledgments
* PyTorch Template Project https://github.com/victoresque/pytorch-template
* Segmentation Models https://github.com/qubvel/segmentation_models
