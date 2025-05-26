# Get Started

## Environment Setup

```bash
conda create -n ai2 python=3.12 -y
conda activate ai2
pip install -r requirements.txt
```

## Data Preprocessing

First get the data ready by unzipping the "Garbage-Classification.zip"

To get the original pictures resized to the same size, run the following command:

```bash
bash preprocess.sh
```

Then you will see resized images in directory "garbage-dataset-resized"

## Training

Baseline model training, run:

```bash
bash run_cnn_baseline.sh
```

Data augment model training, run:

```bash
bash run_cnn_augment_data.sh
```

Structure improve model training, run:

```bash
bash run_cnn_structure_improve.sh
```

Outputs can be seen in directory "output".
