import matplotlib.pyplot as plt

def plot_loss_curves(results, save_path="contrastive_training_loss_comparison.png"):
    """
    Plots training and validation loss curves for multiple models.
    
    Args:
        results (dict): A dictionary where keys are model names and values are tuples:
                        (train_losses, val_losses)
        save_path (str): Path to save the plotted image.
    """
    plt.figure(figsize=(10, 6))
    for name, (train, val) in results.items():
        plt.plot(train, label=f"{name} Train")
        plt.plot(val, label=f"{name} Val", linestyle='--')
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Across Encoders")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()