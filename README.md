# EY24 Competition - Tropical Cyclone Damage Assessment

![image](https://github.com/user-attachments/assets/11fd79d4-efc1-41f8-936e-f3ccb5107142)

(Collaboration with Nicolás Abbate. The implementation was based on his master's thesis and paper.)

This project is focused on assessing damage caused by tropical cyclones using machine learning models, specifically for building segmentation tasks. The implementation is based on the work from Nicolás Abbate's master’s thesis, which provides the foundation for the pre-trained models and approach used in this repository.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Data Preparation: `build_dataset.py`](#data-preparation-build_datasetpy)
  - [Model Execution: `run_model.py`](#model-execution-run_modelpy)
- [Pre-trained Models](#pre-trained-models)

## Project Overview
The goal of this project is to predict building damage using satellite imagery before and after tropical cyclones. The model performs semantic segmentation to detect damaged structures, aiding in rapid assessment and response.

## Requirements
To run the code, you need the following dependencies:

- Python 3.8 or higher
- TensorFlow >= 2.6.0
- NumPy >= 1.19
- Pandas >= 1.2
- OpenCV >= 4.5
- Matplotlib >= 3.4
- Scikit-learn >= 0.24

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Directory Structure

```bash
EY24-main/
├── data/
│   └── submission_data/          # Input data for predictions
├── pre_trained_models/
│   └── building_segmentation_full_model.h5  # Pre-trained model file
├── scripts/
│   ├── EY23_example/
│   ├── EY24_example/
│   ├── build_dataset.py          # Script to build the dataset
│   ├── custom_models.py          # Defines custom model architectures
│   ├── debug.ipynb               # Notebook for debugging
│   ├── generate_submition.py     # Generates submission files
│   ├── merge_models_predictions.py  # Merges predictions from multiple models
│   ├── plot_tools.py             # Utility functions for plotting
│   ├── run_model.py              # Main script to run the model
│   └── utils.py                  # Helper functions
└── README.md                     # Project documentation
```

## Usage

### Data Preparation: `build_dataset.py`
The `build_dataset.py` script is used to prepare the dataset for training or inference. It processes raw satellite images and creates the necessary input format for the model.

**Key Features:**
- Reads raw satellite images (before and after the cyclone).
- Applies preprocessing steps, including resizing and normalization.
- Generates training samples with corresponding labels for segmentation tasks.

**How to Run:**
```bash
python scripts/build_dataset.py --input_dir data/raw_images/ --output_dir data/processed/
```

## Arguments
- `--input_dir`: Directory containing the raw satellite images.
- `--output_dir`: Directory where the processed dataset will be saved.

## Example
```bash
python scripts/build_dataset.py --input_dir data/raw_images/ --output_dir data/processed/
```

### Model Execution: `run_model.py`
The run_model.py script is the main script for running the pre-trained model on the prepared dataset. It loads the model, performs inference, and saves the results.

**Key Features:**
- Loads the pre-trained segmentation model (building_segmentation_full_model.h5).
- Runs inference on the input images and generates segmentation masks.
- Saves the output masks and generates a summary of predictions.

**How to Run:**
```bash
python scripts/run_model.py --model_path pre_trained_models/building_segmentation_full_model.h5 --input_dir data/processed/ --output_dir results/
```

**Arguments:**
- `--model_path`: Path to the pre-trained model file.
- `--input_dir`: Directory containing the preprocessed input data.
- `--output_dir`: Directory where the output segmentation masks will be saved.

**Example**
```bash
python scripts/run_model.py --model_path pre_trained_models/building_segmentation_full_model.h5 --input_dir data/processed/ --output_dir results/
```

**Output:**
- The script saves the segmentation masks as image files in the specified `output_dir`.
- Additionally, a summary CSV file with prediction statistics can be generated.

### Pre-trained Models

The repository includes a pre-trained model for building segmentation:

- `building_segmentation_full_model.h5`: A neural network trained on satellite imagery for detecting building damage.

To use the model, place it in the `pre_trained_models/directory` as shown in the structure.

