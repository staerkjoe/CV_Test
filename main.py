from pyexpat import model
import wandb
import yaml
from pathlib import Path
from src.data_loader import DataLoader
from src.visuals import Visuals
from src.model import YoloModel
import torch
from ultralytics.utils import (SETTINGS)


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
    epochs = config['model']['epochs']
    batch_size = config['model']['batch_size']
    imgsz = config['model']['imgsz']
    wandb_project = config['wandb']['project']
    wandb_run_name = config['wandb']['run_name']
    frozen_layers = config['model']['freeze']

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load Data
    from roboflow import Roboflow
    rf = Roboflow(api_key=config['roboflow']['api_key'])
    project = rf.workspace(config['roboflow']['workspace']).project(config['roboflow']['project'])
    version = project.version(config['roboflow']['version'])
    version.download(config['roboflow']['model_format'])

    # Load Model
    yolo_model = YoloModel(config)
    model = yolo_model.load_model(device=device)  # pass device into load_model
    model.info()    

    # Train model
    model.train(data=data_path, epochs=epochs, imgsz=imgsz, batch=batch_size, save_period=-1, freeze=frozen_layers, exist_ok=True)

    # Initialize W&B
    wandb.init(project=wandb_project, name=wandb_run_name)
    SETTINGS["wandb"] = True

    # Custom Visualization
    visual = Visuals(config, model)
    total_params, trainable_params = visual.count_parameters()
    trainable_param = visual.plot_trainable_parameters(total_params, trainable_params)
    wandb.log({"trainable_parameters_plot": wandb.Image(trainable_param)})

    # Evaluate model (results will also sync to W&B)
    metrics = model.val()
    print(metrics)
    #wandb.log(metrics)

    # upload best model as a W&B artifact
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(model.ckpt_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    main()