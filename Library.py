import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import os

# Utility to create a directory if it does not exist.
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Utility to wrap the savefig function.
def save(filename):
    plt.tight_layout()
    plt.savefig(filename)

# Wrapper over the correlation matrix, using absolute value for the correlation threshold and a uniform palette.
def correlation_matrix(df, threshold=None, **kwargs):
    return sb.heatmap(
        df.corr()[(df.corr() < -threshold) | (threshold < df.corr())] if threshold else df.corr(),
        vmin=-1, vmax=1, center=0,
        cmap=sb.diverging_palette(20, 220, n=200),
        square=True,
        xticklabels=True, yticklabels=True,
        **kwargs
    )

# Display the boxplots for each attribute given in "stats", using "label_attribute" as the cluster label.
def plot_clusters(stats, df, label_attribute):
    def plot_boxplot(a):
        bp = sb.boxplot(x=label_attribute, y=a, data=df)
        bp.set_xlabel("")
        bp.set_ylabel("")
        plt.axhline(y=df[a].mean(axis=0), color = 'r', linestyle = '--')
    plot_matrix(9, 4, numerical_attributes(stats), plot_boxplot)

# Reorder clusters according to the size of the clusters, in descending order.
def reorder_clusters(df, label_attribute):
    reorder = list(df[label_attribute].value_counts().sort_values(ascending=False).index)
    for i, t in enumerate(reorder):
        df.loc[df[label_attribute] == t, label_attribute] = 42 + t
    for i, t in enumerate(reorder):
        df.loc[df[label_attribute] == 42 + t, label_attribute] = i

# Utility function to swap the clustering label inside of a dataset, from label "i" to label "j".
def swap_indices(df, label_attribute, i, j):
    df.loc[df[label_attribute] == i, label_attribute] = 42 + i
    df.loc[df[label_attribute] == j, label_attribute] = 42 + j
    df.loc[df[label_attribute] == 42 + i, label_attribute] = j
    df.loc[df[label_attribute] == 42 + j, label_attribute] = i

# Utility wrapper over subplot function, by iterating over a list of attributes in a dataset given in "attrs".
def plot_matrix(r, c, attrs, f):
    for i, a in enumerate(attrs):
        plt.subplot(r, c, i+1)
        plt.title(a)
        f(a)

# Attributes selectors on the dataset dictionary with the attribute informations.
# All these combinators return a list of strings given a statistics dictionary.

def statistics_attributes(d):
    return [(k, v) for k, v in d.items() if 'hidden' not in v['info']]

def categorical_attributes(d):
    return [k for k, v in statistics_attributes(d) if 'obj' in v['info']]

def numerical_attributes(d):
    return [k for k, v in statistics_attributes(d) if 'num' in v['info'] or 'int' in v['info']]

def percentage_attributes(d):
    return [k for k, v in statistics_attributes(d) if 'percentage' in v['info']]

def essential_attributes(d):
    return [k for k, v in statistics_attributes(d) if 'non-essential' not in v['info']]

def attributes_with(d, p):
    return [k for k, v in statistics_attributes(d) if p in v['info']]

# Fill the missing values in the dataset with the attribute value specified in the 'fillna' declaration, if present.
def fill_nan_values(df, statistics, callback=None):
    for a, v in statistics_attributes(statistics):
        if 'fillna' in v:
            df[a].fillna(v['fillna'], inplace=True)
            if callback:
                callback(df, a)
    return df

# Drop nan values from the attributes specified in the statistics dictionary, unless the 'no-nan-removal' flag is specified.
def drop_nan_values(df, statistics, callback=None):
    for a, v in statistics_attributes(statistics):
        if 'no-nan-removal' not in v['info']:
            df = df[~df[a].isna()]
            if callback:
                callback(df, a)
    return df

# Outlier removal process; given a dataset and a statistics dictionary, remove the outliers while keeping nan values.
# nan values are kept in order to separate concerns with the previous functions.
def drop_outliers(df, statistics, callback=None):
    for a, v in statistics_attributes(statistics):
        if 'outliers' in v:
            df = df[v['outliers'](df[a]) | df[a].isna()]
            if callback:
                callback(df, a)
    return df

# Helper function to deserialize and serialize statistics dictionaries.

# Only serialize strings and integers (e.g.: outlier combinators are not serializable)
def filter_serializable(d):
    return {k:v for k, v in d.items() if type(v) is int or type(v) is list or type(v) is str}

# Serialize statistics into a file.
def serialize_statistics(stats, filename):
    for a, v in stats.items():
        stats[a] = filter_serializable(v)
    with open(filename, 'w') as f:
        f.write(repr(stats))

# Deserialize and retrieve statistics into a file.
def deserialize_statistics(filename):
    with open(filename, 'r') as f:
        stats = eval(f.read())
    return stats
