import gradio as gr
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# https://code.visualstudio.com/docs/datascience/data-science-tutorial
data = pd.read_csv('titanic3.csv')

data.replace('?', np.nan, inplace=True)
data = data.astype({"age": np.float64, "fare": np.float64})


def plot(data: pd.DataFrame, progress=gr.Progress()):
    progress(0)
    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(5, 30))
    sns.violinplot(x="survived", y="age", hue="sex", data=data, ax=axs[0])
    progress(.2)
    sns.pointplot(x="sibsp", y="survived", hue="sex", data=data, ax=axs[1])
    progress(.4)
    sns.pointplot(x="parch", y="survived", hue="sex", data=data, ax=axs[2])
    progress(.6)
    sns.pointplot(x="pclass", y="survived", hue="sex", data=data, ax=axs[3])
    progress(.8)
    sns.violinplot(x="survived", y="fare", hue="sex", data=data, ax=axs[4])
    progress(1)
    plt.savefig("fig1.png")
    plt.close(fig)  # Close the figure to free up memory
    plots = ["fig1.png"]

    data.replace({'male': 1, 'female': 0}, inplace=True)
    # Create new relatives column
    data['relatives'] = data.apply(lambda row: int(
        (row['sibsp'] + row['parch']) > 0), axis=1)
    survived_corr = data.corr(method='pearson', numeric_only=True).abs()[
        ["survived"]]

    # Training Dataset
    # data = data[['sex', 'pclass', 'age','relatives', 'fare', 'survived']].dropna()

    return plots, survived_corr


inputs = [gr.Dataframe(label="Titanic Survivor Data", value=data)]
outputs = [gr.Gallery(label="Profiling Dashboard", height='auto'),
           gr.DataFrame(label="Suvived Correlation")]

gr.Interface(
    fn=plot,
    inputs=inputs,
    outputs=outputs,
    examples=[data.head(5)],
    title="Titanic Survior Analysis Dashboard",
    theme='freddyaboulton/dracula_revamped').launch(debug=True)
