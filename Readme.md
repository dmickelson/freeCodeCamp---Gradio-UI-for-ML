![alt text](/gradio.png)

# Titanic Survivor Analysis Dashboard with Gradio

## Introduction

Adjusting, visualizing, and sharing machine learning models can be much easier with a sleek user interface. Gradio is a Python library that enables you to demo your machine learning model with a user-friendly web interface so that anyone can use it, anywhere.

This repository demonstrates how to use Gradio to create an interactive dashboard for analyzing the Titanic dataset. The dashboard includes various visualizations and a feature correlation analysis to understand the factors influencing survival.

![alt text](/titanic.png)

## Features

- **Interactive Visualizations**: Generate and display plots of different features from the Titanic dataset.
- **Correlation Analysis**: Compute and display the correlation of features with the survival rate.
- **User-Friendly Interface**: Easily interact with the data and visualizations through a web interface.

## Prerequisites

Ensure you have the following installed:

- Python 3.6+
- pandas
- numpy
- seaborn
- matplotlib
- gradio

You can install the required Python packages using the following command:

```
pip install pandas numpy seaborn matplotlib gradio
```

## Usage

#### 1. Clone the Repository.

Clone this repository to your local machine:

```
git clone https://github.com/yourusername/titanic-gradio-dashboard.git
cd titanic-gradio-dashboard
```

#### 2. Download the Titanic Dataset

Ensure you have the Titanic dataset (titanic3.csv) in the root directory of the repository. You can download it from [here](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv).

#### 3. Run the Gradio Interface

Execute the Python script to launch the Gradio interface:

```
python titanic_survivability.py
```

#### 4. Access the Dashboard

Once the script is running, Gradio will provide you with a local web address. Open this address in your web browser to access the Titanic Survivor Analysis Dashboard.

## Code Explanation

Here's a brief overview of what each part of the code does:

- Data Loading and Preprocessing: Load the Titanic dataset, replace missing values, and convert data types for specific columns.
- Plot Function: Generate various plots (violin plots, point plots) to visualize the relationship between features (e.g., age, fare, pclass) and survival. Save these plots as images.
- Correlation Analysis: Compute the correlation between features and survival, ranking them by importance.
- Gradio Interface: Define the inputs (Titanic dataset) and outputs (plots, correlation analysis) for the Gradio interface. Launch the interface with a specified theme.

## Conclusion

This project showcases the power of Gradio in creating interactive web interfaces for data visualization and analysis. By following the steps outlined above, you can easily set up your own Titanic Survivor Analysis Dashboard and explore the data in a user-friendly manner.
