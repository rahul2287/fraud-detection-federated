"""
Utility functions for logging and metrics display.
"""
def log_metrics(metrics):
    print("Loss: {:.4f}, Accuracy: {:.4f}".format(metrics[0], metrics[1]))
