import gradio as gr
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from a CSV file
data = pd.read_csv('titanic3.csv')

# Replace '?' with NaN and convert 'age' and 'fare' columns to float
data.replace('?', np.nan, inplace=True)
data = data.astype({"age": np.float64, "fare": np.float64})


def plot(data: pd.DataFrame, progress=gr.Progress()):
    """
    Generates and saves several plots from the Titanic dataset and calculates the correlation
    between features and survival rate.

    Parameters:
    - data: pd.DataFrame - The Titanic dataset.
    - progress: gr.Progress - A progress tracker for updating the progress of plot generation.

    Returns:
    - plot_files: List of file paths to the saved plots.
    - survived_corr: DataFrame of features correlated with survival, ranked by importance.
    """
    # Initialize progress
    progress(0)
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    plot_files = []

    # Plot 1: Violin plot of age by survived and sex
    fig, axs = plt.subplots(figsize=(10, 5))
    sns.violinplot(x="survived", y="age", hue="sex", data=data, ax=axs)
    filename = os.path.join(output_dir, f"plot_age.png")
    plt.savefig(filename)
    plt.close(fig)
    plot_files.append(filename)
    progress(0.2)

    # Plot 2: Point plot of sibsp by survived and sex
    fig, axs = plt.subplots(figsize=(10, 5))
    sns.pointplot(x="sibsp", y="survived", hue="sex", data=data, ax=axs)
    filename = os.path.join(output_dir, f"plot_sibsp.png")
    plt.savefig(filename)
    plt.close(fig)
    plot_files.append(filename)
    progress(0.4)

    # Plot 3: Point plot of parch by survived and sex
    fig, axs = plt.subplots(figsize=(10, 5))
    sns.pointplot(x="parch", y="survived", hue="sex", data=data, ax=axs)
    filename = os.path.join(output_dir, f"plot_parch.png")
    plt.savefig(filename)
    plt.close(fig)
    plot_files.append(filename)
    progress(0.6)

    # Plot 4: Point plot of pclass by survived and sex
    fig, axs = plt.subplots(figsize=(10, 5))
    sns.pointplot(x="pclass", y="survived", hue="sex", data=data, ax=axs)
    filename = os.path.join(output_dir, f"plot_pclass.png")
    plt.savefig(filename)
    plt.close(fig)
    plot_files.append(filename)
    progress(0.8)

    # Plot 5: Violin plot of fare by survived and sex
    fig, axs = plt.subplots(figsize=(10, 5))
    sns.violinplot(x="survived", y="fare", hue="sex", data=data, ax=axs)
    filename = os.path.join(output_dir, f"plot_fare.png")
    plt.savefig(filename)
    plt.close(fig)
    plot_files.append(filename)
    progress(1)

    # Replace 'male' with 1 and 'female' with 0 in the dataset
    data.replace({'male': 1, 'female': 0}, inplace=True)

    # Create a new column 'relatives' indicating if the passenger has relatives on board
    data['relatives'] = data.apply(lambda row: int(
        (row['sibsp'] + row['parch']) > 0), axis=1)

    # Calculate the correlation between features and survival
    survived_corr = data.corr(method='pearson', numeric_only=True).abs()[
        ["survived"]]
    survived_corr = survived_corr.reset_index().dropna(
    ).sort_values(by="survived", ascending=False)
    survived_corr.columns = ["Feature", "Survived"]

    return plot_files, survived_corr


# Define the Gradio inputs and outputs
inputs = [gr.Dataframe(label="Titanic Survivor Data", value=data)]
outputs = [gr.Gallery(label="Profiling Dashboard", format='png', type="filepath", columns=1, rows=5),
           gr.DataFrame(label="Survived Correlation ranked by Importance")]

# Create and launch the Gradio interface
gr.Interface(
    fn=plot,
    inputs=inputs,
    outputs=outputs,
    examples=[data.head(5)],
    title="Titanic Survivor Analysis Dashboard",
    theme='freddyaboulton/dracula_revamped'
).launch(debug=True)
