import yaml
from pathlib import Path
from src.data_loader import DataLoader

def load_config():
    config_path = Path(__file__).resolve().parent / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
    
def main():
    config = load_config()  # or load_config("path/to/other.yaml")
    print("Loaded config:", config)

    dataloader = DataLoader(config)
    dataset = dataloader.download_dataset()



if __name__ == "__main__":
    main()