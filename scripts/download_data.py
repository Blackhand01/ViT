import requests
from pathlib import Path
import zipfile

# Define the source URL and the destination directory
dataset_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
destination_dir = Path("../data/pizza_steak_sushi")

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
