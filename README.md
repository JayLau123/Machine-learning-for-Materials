# CNN-for-Molecular-Properties-
Trained a convolutional neural network (CNN) for image analysis and pattern recognition with molecular dataset QM9 and toolbox SchNetPack on Google Colab.

Chose atomization energy as the imput and prediction parameter, and 1000 training examples in 2D/3D dataset for validation, successfully reduced the learning rate without improvement of the validation loss. 

Investigated its ‘Final validation MAE’ with 200 epochs on GPU (Tesla T4) and predictions, then repeat with 3D data as descriptors and compared their performance.

Please find more details and code in the link:

https://colab.research.google.com/drive/1jH3dWjqh24FpDSyEQ1d2ynV3I2aEvhif?usp=sharing

# Computation-aided characterization for molecules

Train a forward model with public database from PubChem, NIST, Wiley’s library, etc. Once we input a molecule structure (graph-based representation), it will output the spectra automatically.



# Inverse Problem: Graph theory+ CNN for molecular structure reconstruction(Current project)

Based on graph theory and imitation learning method to train a Convolutional Neural Network, the data we feed to the model are many graphs, which can be regarded as a function of vertex and edges: $G=G(V,E)$. The degree of each vertex $V$ and weights of each edge $E$ represents different atoms and chemical bond. So we start from a simple graph and Markov decision process, by adding or deleting the vertice and edges, with many iterration process until we find the most accurate graph that can describe the chemical structure. 

For more relevant works: http://ericjonas.com/

## Introduction

### Graph-based representation for molecules

Graph neural network(GNN) is very active field nowadays. Its explicit structural relationship and easy-to-model mathematical properties make it increasingly popular. We can recall from early education that any molecular structure contains atoms and chemical bonds, and it is so intuitive to combine it with the graph theory and GNN. 

Therefore, it's reasonable to transfer a chemical molecule into a unique graph as below:

1. Atoms=Vertices $V=\{v_{i}\}$. Different colored vertices represent different kinds of atoms, which corresponding to a particular element. The maximum vertex degree represents the valence of that element.

2. Chemical bonds=Edges $E=\{e_{ij}\}$, which means there is an edge between a pair of vertices $(v_{i}, v_{j}) \in V$. Each edge associated with a weight $w_{i} \in \{w_{1}, w_{2}, w_{3}, w_{4}\}$, corresponding to single, double, triple, and aromatic bonds.

3. Molecules=Graphs $G=(V,E,C)$, where $V$ indicates atoms, $E$ indicates chemical bonds, while $C$ indicates internal physical or chemical constraints.

In measurement technique such as Raman spectroscopy experiments, we only observe spectrum denoted as $P$, it's a type of 1D or 2D data. In our following work, $P$ would be used in loss function to measure the accuracy of the forward model $\boldsymbol{f}$.

After modeling of molecules, we formulate the problem as a Markov decision process(MDP), where we just start with a very simple graph with a collection of vertices $V$, and their observed spectrum $P$, as constraints $C$. Then sequentially add or delete bonds $e_{ij}$ with autoregressive process(AR), until the molecule is connected correctly and satisfy valence constraints or other physical conditions.

During the autoregressive process, $G_{k}=(V, P, E_{k})$ represent the $k$th state of the model, where $k \in [1,2,3...K]$ represent how many existing edges $E_{k}$ in current state. 

As a result, we obtain a graph as a candidate structure $x$, and obove process can be repeated $N$ times to generate $N$ possible candidates $G_{K}^{N}$. Then evaluate the quality of these candidates based on a loss function: probably approximately correct function.





