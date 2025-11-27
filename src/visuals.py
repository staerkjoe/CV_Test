import matplotlib.pyplot as plt
import numpy as np

class Visuals:
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def count_parameters(self):
        """Count total and trainable parameters in the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_params, trainable_params


    def plot_trainable_parameters(self,total_params, trainable_params):
        if isinstance(total_params, dict):
            models = list(total_params.keys())
            totals = np.array([total_params[m] for m in models], dtype=float)
            if isinstance(trainable_params, dict):
                trains = np.array([trainable_params.get(m, 0) for m in models], dtype=float)
            else:
                trains = np.array(trainable_params, dtype=float)
        else:
            # Ensure scalars become 1-D arrays so len() works
            totals = np.atleast_1d(np.array(total_params, dtype=float))
            trains = np.atleast_1d(np.array(trainable_params, dtype=float))
            models = [str(i) for i in range(len(totals))]

        # Convert to millions for display
        scale = 1e6
        totals_M = totals / scale
        trains_M = trains / scale

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.36

        bars_total = ax.bar(x - width/2, totals_M, width, label='Total params', color='#3498db')
        bars_train = ax.bar(x + width/2, trains_M, width, label='Trainable params', color='#2ecc71')

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.set_ylabel('Parameters (M)', fontsize=12)
        ax.set_title('Model Complexity Comparison', fontsize=14, fontweight='bold')
        ax.legend()

        # Add value labels above bars
        def annotate(bars, vals):
            max_h = max(np.max(totals_M) if len(totals_M)>0 else 0, np.max(trains_M) if len(trains_M)>0 else 0)
            offset = max_h * 0.02 if max_h > 0 else 0.01
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + offset,
                        f'{val:.2f}M',
                        ha='center', va='bottom', fontsize=9)

        annotate(bars_total, totals_M)
        annotate(bars_train, trains_M)

        plt.tight_layout()
        plt.close(fig)
        return fig
        