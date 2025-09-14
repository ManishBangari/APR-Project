import kagglehub
import os
from pathlib import Path
import shutil

data_dir = Path("Data")
data_dir.mkdir(exist_ok=True) 

# Download the dataset
dataset_name = "mohammadtalib786/retail-sales-dataset"
download_path = kagglehub.dataset_download(dataset_name)

# Find the CSV file in the downloaded dataset
csv_file = Path(download_path) / "retail_sales_dataset.csv"
if not csv_file.exists():
    raise FileNotFoundError(f"CSV file not found in {download_path}")

# Copy the CSV to the Data directory
target_file = data_dir / "retail_sales_dataset.csv"
shutil.copy(csv_file, target_file)

print(f"Dataset CSV downloaded and saved to: {target_file}")