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
   git clone https://github.com/yourusername/ViT.git
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

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Daniel Bourke's Vision Transformer Tutorial](https://github.com/mrdbourke/pytorch-deep-learning)

## Acknowledgments

- **Daniel Bourke** for his comprehensive [Vision Transformer tutorial](https://github.com/mrdbourke/pytorch-deep-learning).
- The [PyTorch community](https://discuss.pytorch.org/) for continuous support and development.

---

Make sure to install these dependencies to ensure that the project runs smoothly.

If you have any questions or need further assistance, feel free to reach out!
