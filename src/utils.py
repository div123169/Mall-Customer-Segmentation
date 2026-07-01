import os
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

# Define raw URL for Mall_Customers.csv dataset (Kaggle replica)
DATASET_URL = (
    "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/"
    "Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/"
    "Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
)

# Project paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "Mall_Customers.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")
PLOTS_DIR = os.path.join(NOTEBOOKS_DIR, "plots")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")

def init_directories():
    """Initializes all project subdirectories."""
    for folder in [DATASET_DIR, MODELS_DIR, NOTEBOOKS_DIR, PLOTS_DIR, ASSETS_DIR]:
        os.makedirs(folder, exist_ok=True)

def download_dataset(url=DATASET_URL, dest_path=DATASET_PATH):
    """Downloads the Mall Customers dataset if not already present locally."""
    init_directories()
    if not os.path.exists(dest_path):
        print(f"Dataset not found locally. Downloading from {url}...")
        try:
            # Add user-agent header to avoid potential HTTP 403 Forbidden issues
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Dataset downloaded successfully and saved to {dest_path}")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            raise e
    else:
        print(f"Dataset already exists at {dest_path}")

def load_data(file_path=DATASET_PATH):
    """Loads the dataset into a pandas DataFrame, downloading it first if necessary."""
    if not os.path.exists(file_path):
        download_dataset(dest_path=file_path)
    df = pd.read_csv(file_path)
    if 'Genre' in df.columns:
        df = df.rename(columns={'Genre': 'Gender'})
    return df

def save_plot(fig, filename, folder=PLOTS_DIR, dpi=300):
    """Saves a matplotlib/seaborn figure to the specified folder with high resolution."""
    init_directories()
    # Save in specified folder (default notebooks/plots)
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Plot saved to {path}")
    
    # Also mirror inside assets/ folder for Streamlit web app
    assets_path = os.path.join(ASSETS_DIR, filename)
    fig.savefig(assets_path, dpi=dpi, bbox_inches="tight")
    print(f"Plot mirrored to {assets_path}")

# Initialize directories automatically when utils is imported
init_directories()
