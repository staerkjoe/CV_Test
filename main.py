from pyexpat import model
import wandb
import yaml
from pathlib import Path
from src.data_loader import DataLoader
from src.visuals import Visuals
from src.model import YoloModel
import torch
from ultralytics.utils import SETTINGS


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

    # Enable Ultralytics W&B integration
    SETTINGS["wandb"] = True

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
    model = yolo_model.load_model(device=device)
    model.info()

    # Initialize Visuals class
    visual = Visuals(config, model.model)
    
    # Count parameters BEFORE training (store them for later use)
    total_params, trainable_params = visual.count_parameters()
    print(f"\n{'='*50}")
    print(f"MODEL PARAMETERS")
    print(f"{'='*50}")
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Frozen parameters:     {total_params - trainable_params:,}")
    print(f"{'='*50}\n")
    
    # Define custom callback to log all visualizations before W&B finishes
    def on_train_end(trainer):
        """Called when training ends, before W&B closes"""
        print("\n" + "="*60)
        print("GENERATING CUSTOM VISUALIZATIONS & EXPORTS")
        print("="*60)
        
        # Log all visualizations and CSVs
        # Pass the pre-calculated params since trainer.model might be wrapped differently
        visual.log_all_training_visualizations(
            trainer, 
            total_params=total_params,
            trainable_params=trainable_params
        )
        
        # Export per-class metrics from final validation
        print("\n" + "-"*60)
        print("Exporting per-class metrics...")
        print("-"*60)
        
        # Run final validation to get metrics
        metrics = trainer.validator.metrics if hasattr(trainer, 'validator') else None
        if metrics:
            class_metrics_path = Path(trainer.save_dir) / 'class_metrics.csv'
            visual.export_class_metrics_csv(metrics, class_metrics_path)
            
            # Log to W&B
            if class_metrics_path.exists():
                import pandas as pd
                wandb.log({"validation/per_class_metrics": wandb.Table(dataframe=pd.read_csv(class_metrics_path))})
                print(f"Logged per-class metrics to W&B")
        
        # Upload best model as W&B artifact
        print("\n" + "-"*60)
        print("Uploading model artifact...")
        print("-"*60)
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(trainer.best)
        wandb.log_artifact(artifact)
        print(f"Best model saved as artifact: {trainer.best}")
        
        print("\n" + "="*60)
        print("CUSTOM VISUALIZATIONS & EXPORTS COMPLETE")
        print("="*60 + "\n")

    # Add callback to model
    model.add_callback("on_train_end", on_train_end)

    # Train model (Ultralytics will manage W&B automatically)
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    results = model.train(
        data=data_path, 
        epochs=epochs, 
        imgsz=imgsz, 
        batch=batch_size, 
        save_period=-1, 
        freeze=frozen_layers, 
        exist_ok=True,
        project=wandb_project,
        name=wandb_run_name
    )

    # Evaluate model (metrics auto-logged to W&B by Ultralytics)
    print("\n" + "="*60)
    print("RUNNING FINAL VALIDATION")
    print("="*60 + "\n")
    
    metrics = model.val()
    print(metrics)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()