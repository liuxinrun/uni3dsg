## Uni3DSG: Unifying 3D Perception and Reasoning for End-to-End Scene Graph Generation

This repository contains an implementation of Uni3DSG, which unifies the 3D object detection and scene graph generation.

### Installation

For convenience, we provide a [requirement.txt](requirement.txt).
This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework `v1.1.0`. If not using Docker, please follow [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/get_started.md) for the installation instructions.

### Getting Started

Please see [test_train.md](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/user_guides/train_test.md) for some basic usage examples.

#### Data Preprocessing

Uni3DSG is trained and tested on [3RScan](data/3rscan) for 3D object detection, and [3DSSG](data/3rscan) for 3d scene graph generation.

Alternatively, you can preprocess the data by youself. 

For the 3Rscan datasets, you can either download pre-computed superpoints at UniDet3D [Hugging Face](https://huggingface.co/datasets/maksimko123/UniDet3D), or compute them using [superpoint_transformer](https://github.com/drprojects/superpoint_transformer).

For the 3DSSG datasets, please make sure you agree the 3RScan Terms of Use first, and get the download script and put it right at the 3RScan main directory. Then run

```bash
python scripts/RUN_prepare_dataset_3RScan.py --download --thread 8
```

#### Training

Before training, please download the backbone [checkpoint](https://github.com/filapro/oneformer3d/releases/download/v1.0/oneformer3d_1xb4_scannet.pth) and save it under `work_dirs/tmp`.

To train Two-Stage on 3DSSG dataset, simply run the [training](tools/train.py) script:

```bash
python tools/train.py configs/two_stage_3rscan.py
```

To train Uni3DSG on 3DSSG dataset, simply run the [training](tools/train.py) script:

```bash
python tools/train.py configs/uni3dsg_3rscan.py
```

#### Testing

To test a trained model, you can run the [testing](tools/test.py) script:

```bash
python tools/test.py configs/two_stage_3rscan.py \
    work_dirs/two_stage_3rscan/epoch_512.pth
```
or
```bash
python tools/test.py configs/uni3dsg_3rscan.py \
    work_dirs/uni3dsg_3rscan/epoch_512.pth
```