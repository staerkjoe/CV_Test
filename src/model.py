# ...existing code...
class YoloModel:
    def __init__(self, config):
        self.config = config
        self.model_name = self.config['model']['name']
        self.model_format = self.config['roboflow']['model_format']
        self.model = None

    def load_model(self, device=None):
        """Load YOLO model and keep reference in self.model."""
        if self.model_format == "yolov8":
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            if device:
                # optional: move underlying torch model to device
                try:
                    self.model.to(device)
                except Exception:
                    pass
            return self.model
        else:
            raise ValueError(f"Unsupported model format: {self.model_format}")

    