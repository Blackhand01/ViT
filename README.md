# Vision Transformer (ViT) from Scratch

This repository contains an implementation of the Vision Transformer (ViT) architecture from scratch using PyTorch. The project replicates the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) and applies it to a custom image classification task involving pizza, steak, and sushi images.

## Project Structure

```
ViT_project/
├── data/
│   └── pizza_steak_sushi/
│       ├── train/
│       └── test/
├── models/
├── notebooks/
│   └── ViT_from_scratch.ipynb
├── scripts/
│   ├── data_setup.py
│   ├── engine.py
│   ├── model_builder.py
│   ├── predictions.py
│   ├── train.py
│   └── utils.py
├── README.md
└── requirements.txt
```

- **data/**: Contains the dataset used for training and testing.
- **models/**: Stores trained model checkpoints.
- **notebooks/**: Jupyter notebooks for exploration and development.
- **scripts/**: Python scripts for data loading, model definition, training, and prediction.
- **README.md**: Project documentation.
- **requirements.txt**: Python dependencies required to run the project.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.12 or higher
- torchvision 0.13 or higher

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/ViT_project.git
   cd ViT_project
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. **Download the dataset:**

   The dataset should be placed in the `data/` directory. Ensure that the structure is as follows:

   ```
   data/
   └── pizza_steak_sushi/
       ├── train/
       │   ├── pizza/
       │   ├── steak/
       │   └── sushi/
       └── test/
           ├── pizza/
           ├── steak/
           └── sushi/
   ```

   You can download the dataset using the following script:

   ```python
   # download_data.py
   import requests
   from pathlib import Path
   import zipfile

   # Define the source URL and the destination directory
   dataset_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
   destination_dir = Path("data")

   # Download the dataset
   if not (destination_dir / "pizza_steak_sushi").exists():
       destination_dir.mkdir(parents=True, exist_ok=True)
       zip_path = destination_dir / "pizza_steak_sushi.zip"
       with open(zip_path, "wb") as f:
           print("Downloading dataset...")
           response = requests.get(dataset_url)
           f.write(response.content)
       # Unzip the dataset
       with zipfile.ZipFile(zip_path, "r") as zip_ref:
           print("Extracting dataset...")
           zip_ref.extractall(destination_dir)
       zip_path.unlink()
       print("Dataset ready!")
   else:
       print("Dataset already exists.")
   ```

   Save this script as `download_data.py` and run:

   ```bash
   python download_data.py
   ```

### Training the Model

Navigate to the `scripts/` directory and run the `train.py` script:

```bash
cd scripts
python train.py --train_dir ../data/pizza_steak_sushi/train --test_dir ../data/pizza_steak_sushi/test
```

You can adjust hyperparameters using command-line arguments. For example:

```bash
python train.py --epochs 20 --learning_rate 0.001 --batch_size 64
```

For a full list of arguments, run:

```bash
python train.py --help
```

### Making Predictions

Use the `predictions.py` script to make predictions on new images:

```bash
python predictions.py --image_path ../path_to_your_image.jpg --model_path ../models/vit_model.pth
```

Ensure that you update the `image_path` and `model_path` accordingly.

### Exploring with Jupyter Notebook

For an interactive exploration, open the notebook:

```bash
cd notebooks
jupyter notebook ViT_from_scratch.ipynb
```

## Requirements

All required packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Daniel Bourke's Vision Transformer Tutorial](https://github.com/mrdbourke/pytorch-deep-learning)

## Acknowledgments

- **Daniel Bourke** for his comprehensive [Vision Transformer tutorial](https://github.com/mrdbourke/pytorch-deep-learning).
- The [PyTorch community](https://discuss.pytorch.org/) for continuous support and development.

---

### **requirements.txt**

```
torch>=1.12
torchvision>=0.13
torchinfo>=1.5.4
matplotlib>=3.3.0
numpy>=1.18.0
Pillow>=7.0.0
tqdm>=4.0.0
requests>=2.0.0
```

---

Make sure to install these dependencies to ensure that the project runs smoothly.

If you have any questions or need further assistance, feel free to reach out!