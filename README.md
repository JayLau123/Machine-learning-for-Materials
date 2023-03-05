# CNN-for-Molecular-Properties-
Trained a convolutional neural network (CNN) for image analysis and pattern recognition with molecular dataset QM9 and toolbox SchNetPack on Google Colab.

Chose atomization energy as the imput and prediction parameter, and 1000 training examples in 2D/3D dataset for validation, successfully reduced the learning rate without improvement of the validation loss. 

Investigated its ‘Final validation MAE’ with 200 epochs on GPU (Tesla T4) and predictions, then repeat with 3D data as descriptors and compared their performance.

Please find more details and code in the link:

https://colab.research.google.com/drive/1jH3dWjqh24FpDSyEQ1d2ynV3I2aEvhif?usp=sharing

# Inverse Problem: Graph theory+ CNN for molecular structure reconstruction

Based on graph theory and imitation learning method to train a Convolutional Neural Network, the data we feed to the model are many graphs, which can be regarded as a function of vertex and edges: $G=G(V,E)$. The degree of each vertex $V$ and weights of each edge $E$ represents different atoms and chemical bond. So we start from a simple graph and Markov decision process, by adding or deleting the vertice and edges, with many iterration process until we find the most accurate graph that can describe the chemical structure. 

For more relevant works: http://ericjonas.com/
