# R-LiViT Evaluation: LiDAR
**Note**: This repository is dedicated to the LiDAR benchmark component of the R-LiViT dataset and serves as a submodule of the entire R-LiViT project repository.

This project builds upon the [PCDet](https://github.com/open-mmlab/OpenPCDet) framework, integrating our custom dataset to evaluate various LiDAR detectors. Additionally, we have implemented and integrated three state-of-the-art 3D Multi-Object Trackers: AB3DMOT, Mahalanobis Tracker, and SimpleTrack.

## Features

- **Benchmarking Object Detectors**: Benchmark all the models considered in the paper (PointPillars, PointRCNN, PV-RCNN) in object detection on the R-LiViT dataset in the LiDAR modality. 
- **Benchmarking Trackers**: Benchmark all the methods considered in the paper (AB3DMOT, SimpleTrack, Mahalanobis) in object tracking on the R-LiViT dataset in the LiDAR modality. 

Additional features:

- **R-LiViT PCDet Integration**: Includes R-LiViT dataset integration into the PCDet framework.
- **Extending the Pipeline**: Pipeline can be easily extended to include additional datasets and models.

## Preparation

### Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install PCDet**:
   Follow the [official PCDet documentation](https://github.com/open-mmlab/OpenPCDet) for detailed installation instructions.

3. **Include Third-party Repos**:
   Clone the required (and wanted) third party repos that are required to reproduce the results from the paper (e.g. [ab3dmot](https://github.com/AIR-THU/DAIR-V2X), [mahalanobis_3d_mot](https://github.com/valeoai/valeo4cast), [mot_3d](https://github.com/tusen-ai/SimpleTrack)) into `third_party/`.

### Dataset Preparation

1. **Dataset Formatting**: Ensure your dataset is formatted to be compatible with the PCDet framework.
2. **Dataset Placement**: Place the formatted dataset within the `data/` directory.

### Model Configuration

1. **Configuration Setup**: Configure your model settings in the `tools/cfgs` directory.
2. **Customization**: Modify the configuration files to align with your dataset specifications and project requirements.

## Training and Benchmarking

### LiDAR 3D Detection Benchmark

1. **Dataset Processing**:
```bash
python -m pcdet.datasets.r_livit.r_livit_dataset create_r_livit_infos tools/cfgs/dataset_configs/r_livit_dataset.yaml
```

2. **Model Configuration**: Edit the model configuration file located in `tools/cfgs` to suit your needs.

3. **Execute Training**:
```bash
python ./tools/train.py --cfg_file tools/cfgs/r_livit_models/pointpillar.yaml
```

4. **Run Detection Benchmarking**:
```bash
sh scripts/benchmark.sh
```

### LiDAR 3D MOT Benchmark

1. **Save Detection Results**:
Use `inference.py` to save the detection results:
```bash
python inference.py --cfg_file tools/cfgs/your_model_config.yaml --data_path data/your_dataset/ --ckpt checkpoints/your_model.pth --save_path results/
```

2. **Run Tracking**:
Execute the 3D MOT benchmark script using the saved detection results:
```bash
sh scripts/trk_r_livit_inference_ab3dmot_cyclist.sh
```

## Results

To test and reproduce the evaluation results using a pretrained checkpoint, follow these steps:

1. **Download Pretrained Checkpoint**:
   Download the pretrained model checkpoint from the provided link and place it in the `checkpoints/` directory.

2. **Run Inference**:
   Use the pretrained checkpoint to run inference on your dataset:
```bash
python inference.py --cfg_file tools/cfgs/your_model_config.yaml --data_path data/your_dataset/ --ckpt checkpoints/your_model.pth --save_path results/
```

3. **Evaluate Results**:
   Use the evaluation script to assess the performance of the model:
```bash
python ./tools/test.py --cfg_file ./tools/cfgs/r_livit_models/pointpillar.yaml --ckpt ./path_to_your_checkpoint.pth --batch_size 1
```
