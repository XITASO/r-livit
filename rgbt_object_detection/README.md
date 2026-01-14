# R-LiViT Evaluation: RGB-T
**Note**: This repository is dedicated to the RGB-T benchmark component of the R-LiViT dataset and serves as a submodule of the entire R-LiViT project repository.

## Features

- **Benchmarking Object Detectors**: Benchmark all the models considered in the paper (Faster R-CNN, YOLOv8, RT-DETR) in object detection on the R-LiViT dataset in the RGB, Thermal, and RGB-T modality. 

Additional features:

- **R-LiViT Dataloaders**: Includes R-LiViT dataloader at `data/rlivit.py`.
- **Support for Further RGB-T Datasets**: Can also be used with other RGB-T datasets mentioned in the paper, such as KAIST, Flir-aligned, LLVIP, M3FD, and SMOD. You can perform similar benchmarking on these datasets.
- **Support for YOLO Variants**: The pipeline supports further YOLO versions implemented in Ultralytics. Be aware that fine-tuning parameters might not be optimal for these versions, as default parameters for YOLOv8 are implemented.
- **Extending the Pipeline**: The pipeline can be easily extended to include additional datasets and models. To do so, wrap them in the model interface defined in `models/model_interface.py` and the data interface in `data/data_interface.py`.

## Preparation

### Installation

#### Running without Docker i.e. in Virtual Environment 

Intall dependencies:

```bash
pip install -r requirements.txt
```

**OR**

#### Using Docker

1. **Build the Docker Image**

   Use the following command to build the Docker image. Replace `PATH_TO_PROJECT` with the path to your `r-livit_evaluation/rgbt_object_detection` directory:

   ```bash
   docker build -t rgbt_object_detection --build-arg USERID=$(id -u) PATH_TO_PROJECT
   ```

2. **Run the Docker Container**

   To run the container and bind your project and dataset directories, use the following command:

   ```bash
   docker run --rm \
     --mount type=bind,source="PATH_TO_PROJECT",target=/workspace \
     --mount type=bind,source="PATH_TO_DATASETS",target="/workspace/data/data_src/" \
     rgbt_object_detection
   ```

   - Replace `PATH_TO_PROJECT` with the absolute path to your local project directory.
   - Replace `PATH_TO_DATASETS` with the absolute path to the directory where your datasets are stored.

   Either run the container interactively by using the `-it` flags, or configure the appropriate startup command directly in the Dockerfile.

### Dataset Preparation

1. **Download Dataset(s)**: Obtain the required datasets simply in their original provided format.
2. **Specify Dataset Paths**: Set their locations in `conf/data.yaml`. The current version assumes you are mounting a folder `data` where all the datasest are placed in onto `/workspace/data/data_src/`. However, you can also just adjust the paths accordingly.

## Training and Benchmarking

### Run Fine-Tuning

To fine-tune the model, execute the `finetuning.py` script accordingly, for example like this:

```bash
python finetuning.py --dataset RLiViT --model YOLOv8 --model_size M --modality rgb --seed 42 --name finetune_1
```
- Refer to the finetuning script for more detailed instructions.

### Run Detection Benchmarking

To benchmark the models, update the script with the model(s), weights, and dataset settings before running it:

```bash
python benchmarking.py
```
- Refer to the benchmarking script for more detailed instructions.

### Use the model weights of our evaluation
We refer to [the zenodo repository](https://doi.org/10.5281/zenodo.18242742) in which the weights of the RGB and IR models are available.
