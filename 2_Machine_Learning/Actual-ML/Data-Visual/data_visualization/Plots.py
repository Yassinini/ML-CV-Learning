#Demo for showing plots progress
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

def sort_by_column(df, column):
    """Sorts a DataFrame by a specified column."""
    return df.sort_values(column)

def remove_nan_rows(df):
    """Removes rows with any NaN values from a DataFrame."""
    return df.dropna()

def scatter(df, x, y, outliers=True, percentile=95, title=""):
    """
    Generates a scatter plot.
    Optionally removes outliers based on a percentile.
    """
    data = remove_nan_rows(df)
    if outliers and pd.api.types.is_numeric_dtype(data[x]) and pd.api.types.is_numeric_dtype(data[y]):
        data = data[(data[x] < np.percentile(data[x], percentile)) & 
                    (data[y] < np.percentile(data[y], percentile))]
    plt.figure(figsize=(12,8))
    plt.scatter(data[x], data[y], alpha=0.2)
    plt.title(title if title 
              else f"{x} to {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.show()

def bar(df, x, y, title=""):
    """Generates a bar plot."""
    data = remove_nan_rows(df)
    plt.figure(figsize=(15,10))
    plt.bar(data[x], data[y])
    plt.title(title if title 
              else f"{x} to {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.show()

def line(df, x, y, outliers=True, percentile=95 , title=""):
    """
    Generates a line plot.
    Optionally removes outliers based on a percentile.
    """
    data = remove_nan_rows(df)
    if outliers and pd.api.types.is_numeric_dtype(data[x]) and pd.api.types.is_numeric_dtype(data[y]):
        data = data[(data[x] < np.percentile(data[x], percentile)) & 
                    (data[y] < np.percentile(data[y], percentile))]
    plt.figure(figsize=(15,10))
    plt.plot(data[x], data[y])
    plt.title(title if title 
              else f"{x} to {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.show()

def countplot(df, x, hue):
    """Generates a seaborn count plot."""
    sns.countplot(x= x,hue= hue, data=df)
    plt.show()

def heatmap(df, cmap):
    """Generates a seaborn heatmap of the correlation matrix for numeric columns."""
    numeric_cols = df.select_dtypes(include='number')
    correlation_matrix = numeric_cols.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, linewidths=0.5)
    plt.show()

def pairplot(df, hue):
    """Generates a seaborn pair plot."""
    sns.pairplot(df , hue=hue)
    plt.show()

def plot_pie_chart_from_category(df, category_column, title="", angle=0):
    """
    Generates a pie chart showing the distribution of a categorical column.
    The slices represent the value counts of the specified category_column.
    """
    value_counts = df[category_column].value_counts()
    plt.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=angle)
    plt.title(title if title else f"Pie chart of {category_column}")
    plt.show()

def barh(df, x, y, title=""):
    """Generates a horizontal bar plot."""
    plt.barh(df[x], df[y])
    plt.title(title if title else f"{x} to {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def hist(df, x , title="", color="blue", edgecolor="black"):
    """Generates a histogram."""
    plt.hist(df[x], color=color, edgecolor=edgecolor)
    plt.title(title if title else f"{x} histogram")
    plt.xlabel(x)
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    pass