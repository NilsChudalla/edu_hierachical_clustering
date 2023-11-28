# Import necessary libraries
import streamlit as st
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import colorcet as cc

# plotting function of debdrogram obtained from # Source: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

def load_toy_dataset(dataset_name='iris'):
    """
    Load a toy dataset from scikit-learn.

    Parameters:
    - dataset_name (str): Name of the dataset to load ('iris', 'blobs', or 'stars').

    Returns:
    - data (numpy.ndarray): The feature data.
    - target (numpy.ndarray): The target labels.
    """

    if dataset_name == 'iris':
        data = datasets.load_iris().data
        target = datasets.load_iris().target
    elif dataset_name == 'blobs':
        data, target = datasets.make_blobs(n_samples=300, cluster_std=2, centers=3, random_state=42)
    elif dataset_name == 'moons':
        data, target = datasets.make_moons(n_samples=300, random_state=42)

    return data, target

def plot_dataset(data, target, dataset_name):
    """
    Plot the selected dataset using Matplotlib.

    Parameters:
    - data (numpy.ndarray): The feature data.
    - target (numpy.ndarray): The target labels.
    - dataset_name (str): Name of the dataset.
    """

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    norm = matplotlib.colors.Normalize(vmin=np.min(target), vmax=np.max(target))
    
    cmap = matplotlib.cm.get_cmap('viridis')
       
    ax1.scatter(data[:, 0],data[:, 1], c=target)
    ax1.set_title(f"{dataset_name} Dataset")
    ax1.set_xlabel("Feature 0")
    ax1.set_ylabel("Feature 1")
    for i in np.unique(target):
        ax1.scatter([],[], color=cmap(norm(i)), label=str(i))
    ax1.legend(title='Target')
    ax1.axis('equal')
    st.pyplot(fig1)

def plot_dendrogram(model, p):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    
    fig2, ax2=plt.subplots(figsize=(8,6))

    ax2.set_title("Hierarchical Clustering Dendrogram")
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, truncate_mode="level", p=p, ax=ax2)#, color_threshold=np.max())
    ax2.set_ylabel("Distance between groups")
    ax2.set_xlabel("Number of points in node (or index of point if no parenthesis).")
    ax2.axis('equal')
    st.pyplot(fig2)
    
def plot_agglo_distance(data,d, linkage):

    model1 = AgglomerativeClustering(n_clusters=None,
                                     distance_threshold=d,
                                     linkage=linkage)
    clusters = model1.fit_predict(data)

    fig3, ax3 = plt.subplots(figsize=(8,6))
    ax3.scatter(data[:,0], data[:,1], c=clusters, cmap=cc.cm.glasbey)
    ax3.set_xlabel('Feature 0')
    ax3.set_ylabel('Feature 1')
    ax3.axis('equal')
    st.pyplot(fig3)


# Use caching for expensive function calls
@st.cache
def expensive_computation():
    # some code
    return 42

# Set the title of your app

st.markdown('This WebApp was developed by CG3 at RWTH Aachen University (Nils Chudalla). It was developed for educational purposes only. Please reference accordingly.')
st.title("Agglomerative clustering")
st.markdown('Use this webapp to perform agglomerative clustering on different classification toy datasets. We use Agglomerative clustering by sci-kit learn. The first interactive widget lets you to select a linkage type. The second widget allows you to control the maximum distance for the clustering algorithm. The resulting clusters are then plotted. The third widget lets you truncate the dendrogram (source: scikit-learn).')
st.markdown('''
            The agglomerative clustering algorithm generally operates as follows:
            ''')
st.markdown('''
            0. Treat every unique object as a single group/cluster.
            1. Compute pair-wise distance between all groups.
            2. Merge TWO most similar clusters (depends on linkage)
            3. Compute distances between new groups.
            4. Repeat 2. until only one cluster is left. 
            ''')
# Add a dropdown to select the dataset
st.subheader('Select dataset')
selected_dataset = st.selectbox("Select a dataset", ['blobs', 'iris', 'moons'] )

# Load the selected dataset
data, target = load_toy_dataset(selected_dataset)

df = pd.DataFrame(data, columns=np.arange(data.shape[1]))
df['target'] = target
st.write(f"Loaded {selected_dataset.upper()} dataset")

if selected_dataset == 'iris':
    st.markdown('''
                The features correspond with the following observations: 
                - 0 = sepal length in cm
                - 1 = sepal width in cm
                - 2 = petal length in cm
                - 3 = petal width in cm
                ''')
    st.markdown('''
                The values of the target correspond with the following species: 
                - 0 = iris setosa
                - 1 = iris versicolor
                - 2 = iris veriginica
''')

# Display some information about the loaded dataset

st.write(f"Number of samples: {data.shape[0]}")
st.write(f"Number of features: {data.shape[1]}")
st.write(f"Number of classes: {len(set(target))}")


# Display the loaded data and target
st.write("Loaded dataset:")
st.write(df)

selection = np.arange(data.shape[0])
np.random.shuffle(selection)

sel = st.slider(
    label='Reduce selection',
    min_value=0,
    max_value=data.shape[0]-1,
    value=data.shape[0]-1,
    step=1
)

data = data[:sel, :]
target = target[:sel]

plot_dataset(data, target, selected_dataset)

st.subheader('Select linkage')
selected_linkage = st.selectbox("Select a linkage type", ['single', 'complete', 'average', 'ward'])
st.markdown('Optimization criterion is:')

if selected_linkage == 'single':
    st.latex(r'''
D_{min}(C_{i}, C_{j}) = \begin{matrix}min\\x\epsilon C_{i}, y\epsilon C_{j}\end{matrix}\left \| x-y \right \|^{2}
               ''')
if selected_linkage == 'complete':
    st.latex(r'''
D_{max}(C_{i}, C_{j}) = \begin{matrix}max\\x\epsilon C_{i}, y\epsilon C_{j}\end{matrix}\left \| x-y \right \|^{2}
               ''')
    
if selected_linkage == 'average':
    st.latex(r'''
D_{average}(C_{i}, C_{j}) = \begin{matrix}max\\x\epsilon C_{i}, y\epsilon C_{j}\end{matrix}\left \| x-y \right \|^{2}
               ''')
        

model = AgglomerativeClustering(distance_threshold=0, linkage=selected_linkage, n_clusters=None)
model = model.fit(data[:,:2])

st.subheader('Select maximum distance for cluster merging')

d = st.slider(
    label='Maximum distance',
    min_value=0.0,
    max_value=np.max(model.distances_)*1.01,
    value=np.max(model.distances_)*1.01,
    step=np.max(model.distances_/100)
)

plot_agglo_distance(data[:,:2], d, selected_linkage)

st.subheader('Select depth of dendrogram')
p = st.slider(
        label="Set parameter 'p' to truncate dendrogram",
        value=3,
        min_value=1,
        max_value=8,
        step=1)
plot_dendrogram(model, p)


# Define data for the table
table_data = [
    ["Linkage type", "Property", "Monotony", "Proximity metric", "Comment"],
    ["Single", "contracting", "Yes", "all", "results in chains"],
    ["Complete", "dilating", "Yes", "all", "results in small groups"],
    ["Average", "conservative", "Yes", "all", "-"],
    ["Ward", "conservative", "No", "Distance metrics", "results in equally sized groups"],
]

# Create a Markdown table
markdown_table = "|".join(table_data[0]) + "\n" + "|".join(["---"] * 5) + "\n"
for row in table_data[1:]:
    markdown_table += "|".join(row) + "\n"

# Display the Markdown table
st.markdown(markdown_table)
