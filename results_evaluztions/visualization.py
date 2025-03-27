import pandas as pd
import matplotlib

matplotlib.use('TkAgg')  # Set the interactive backend
import matplotlib.pyplot as plt
import time


def live_training_visualization(csv_file='log.csv', refresh_interval=1):
    """
    Live visualization of the training process from a CSV log file.

    Parameters:
        csv_file (str): Path to the CSV file containing logs.
        refresh_interval (int): Time interval (in seconds) to refresh the plot.
    """
    plt.ion()  # Turn on interactive mode for live updates

    # Initialize the figure and axes
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Live Training Visualization", fontsize=16)

    # Define the metrics to plot
    metrics = {
        "Loss": ["TRAINING_LOSS", "FORGET_LOSS", "VAL_LOSS"],
        "Accuracy": ["TRAINING_ACC", "FORGET_ACC", "VAL_ACC"]
    }

    lines = {}  # Dictionary to store line objects for each metric

    while True:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Clear previous plots
            for axis in ax.flatten():
                axis.clear()

            # Plot Losses
            ax[0, 0].set_title("Training Loss")
            ax[0, 0].set_xlabel("Epochs")
            ax[0, 0].set_ylabel("Loss")
            for metric in metrics["Loss"]:
                if metric in df.columns:
                    lines[metric], = ax[0, 0].plot(df[metric], label=metric)
            ax[0, 0].legend()

            # Plot Accuracies
            ax[0, 1].set_title("Training Accuracy")
            ax[0, 1].set_xlabel("Epochs")
            ax[0, 1].set_ylabel("Accuracy")
            for metric in metrics["Accuracy"]:
                if metric in df.columns:
                    lines[metric], = ax[0, 1].plot(df[metric], label=metric)
            ax[0, 1].legend()

            # Pause to allow the plot to update
            plt.pause(refresh_interval)

        except KeyboardInterrupt:
            print("Visualization stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(refresh_interval)


# Call the function to start live visualization
import sys

csv_file = sys.argv[1]
live_training_visualization(csv_file=csv_file, refresh_interval=2)