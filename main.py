import yaml
from pathlib import Path
from src.data_loader import DataLoader
from src.model import Model
import torch

def load_config():
    config_path = Path(__file__).resolve().parent / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
    
def main():
    config = load_config()
    print("Loaded config:", config)
    data_path = config['data']['path']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataloader = DataLoader(config)
    dataloader.download_dataset()
    dataloader = dataloader  # ...existing code...

    model = Model(config)   # pass device into Model
    model = model.load_model(device=device)  # pass device into load_model
    print(model.info())
    model.train(data=data_path, epochs=config['model']['epochs'], imgsz=config['model']['img_size'])
  

if __name__ == "__main__":
    main()