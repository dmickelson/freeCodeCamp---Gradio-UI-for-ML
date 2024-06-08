import gradio as gr
import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt

# https://code.visualstudio.com/docs/datascience/data-science-tutorial
data = pd.read_csv('titanic3.csv')

data.replace('?', np.nan, inplace=True)
data = data.astype({"age": np.float64, "fare": np.float64})


def plot(data: pd.DataFrame, progress=gr.Progress()):
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

    data.replace({'male': 1, 'female': 0}, inplace=True)
    # Create new relatives column
    data['relatives'] = data.apply(lambda row: int(
        (row['sibsp'] + row['parch']) > 0), axis=1)
    survived_corr = data.corr(method='pearson', numeric_only=True).abs()[
        ["survived"]]

    survived_corr = survived_corr.reset_index().dropna(
    ).sort_values(by="survived", ascending=False)
    survived_corr.columns = ["Feature", "Survived"]

    # Training Dataset
    # data = data[['sex', 'pclass', 'age','relatives', 'fare', 'survived']].dropna()

    return plot_files, survived_corr


inputs = [gr.Dataframe(label="Titanic Survivor Data", value=data)]
outputs = [gr.Gallery(label="Profiling Dashboard", format='png', type="filepath", columns=1, rows=5),
           gr.DataFrame(label="Suvived Correlation ranked by Importance")]

gr.Interface(
    fn=plot,
    inputs=inputs,
    outputs=outputs,
    examples=[data.head(5)],
    title="Titanic Survior Analysis Dashboard",
    theme='freddyaboulton/dracula_revamped').launch(debug=True)
