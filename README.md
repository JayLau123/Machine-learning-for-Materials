## Regression model

Probably approximately correct function(PAC) explains how to find the best candidate structure $x$ with a high probability, which suits for loss function in our model.
 
$$p(|f(x)-y| \leq \epsilon) \geq 1-\delta$$

$x$ is the candidate structure, $y$ is the observed properties $P$ from spectrum, which is the true label. If the prediction is very accurate, then $|f(x)-P|\leq \epsilon$. We can calculate the spectrum properties with known atoms and their structure based on quantum chemistry. Let's denote it as $\boldsymbol{f}$. We hope to find the best candidate structure $x$ with a high probability: $p(|f(x)-P| \leq \epsilon) \geq 1-\delta$. However, regardless of the difficulties to choose exchange-correlation functional $E_{XC}$ in solving Kohn-Sham equation(KS), the $\boldsymbol{f}$ can be very computational-expensive. Alternatively, can we use current database from Materials Project, NOMAD, NIST, etc, to train a graph neural network(GNN) to predict spectrum, so that given a molecular structure (graph-based representation), it can output the spectrum automatically? We are going to build such a network named forward model $\boldsymbol{f}$.


### My personal project:

### Inverse Problem: Graph theory+ CNN for molecular structure reconstruction(Current project)

Based on graph theory and imitation learning method to train a Convolutional Neural Network, the data we feed to the model are many graphs, which can be regarded as a function of vertex and edges: $G=G(V,E)$. The degree of each vertex $V$ and weights of each edge $E$ represents different atoms and chemical bond. So we start from a simple graph and Markov decision process, by adding or deleting the vertice and edges, with many iterration process until we find the most accurate graph that can describe the chemical structure. 

For more relevant works: http://ericjonas.com/

## Introduction

### Graph-based representation for molecules

Graph-based deep learning frameworks have already demonstrated their creative roles in the design and discovery of functional materials by identifying structure–property correlations and making efficient low-cost predictions, by representing material systems in graphs and properly designing message passing strategies. Its explicit structural relationship and easy-to-model mathematical properties make it increasingly popular. We can recall from early education that any molecular structure contains atoms and chemical bonds, and it is so intuitive to combine it with the graph theory and GNN. 

GNN based graph embeddings have recently attracted growing attention, due to the natural representation of molecular structures as graphs. Therefore, it's reasonable to transfer a chemical molecule into a unique graph as below:

1. Atoms=Vertices $V=\{v_{i}\}$. Different colored vertices represent different kinds of atoms, which corresponding to a particular element. The maximum vertex degree represents the valence of that element.

2. Chemical bonds=Edges $E=\{e_{ij}\}$, which means there is an edge between a pair of vertices $(v_{i}, v_{j}) \in V$. Each edge associated with a weight $w_{i} \in \{w_{1}, w_{2}, w_{3}, w_{4}\}$, corresponding to single, double, triple, and aromatic bonds.

3. Molecules=Graphs $G=(V,E,C)$, where $V$ indicates atoms, $E$ indicates chemical bonds, while $C$ indicates internal physical or chemical constraints.

In measurement technique such as Raman spectroscopy experiments, we only observe spectrum denoted as $P$, it's a type of 1D or 2D data. In our following work, $P$ would be used in loss function to measure the accuracy of the forward model $\boldsymbol{f}$.

After modeling of molecules, we formulate the problem as a Markov decision process(MDP), where we just start with a very simple graph with a collection of vertices $V$, and their observed spectrum $P$, as constraints $C$. Then sequentially add or delete bonds $e_{ij}$ with autoregressive process(AR), until the molecule is connected correctly and satisfy valence constraints or other physical conditions.

During the autoregressive process, $G_{k}=(V, P, E_{k})$ represent the $k$th state of the model, where $k \in [1,2,3...K]$ represent how many existing edges $E_{k}$ in current state. 

As a result, we obtain a graph as a candidate structure $x$, and obove process can be repeated $N$ times to generate $N$ possible candidates $G_{K}^{N}$. Then evaluate the quality of these candidates based on a loss function: probably approximately correct function.


Trained a convolutional neural network (CNN) for image analysis and pattern recognition with molecular dataset QM9 and toolbox SchNetPack on Google Colab.

We have achieved so far: 

- Chose atomization energy as the imput and prediction parameter, and 1000 training examples in 2D/3D dataset for validation, successfully reduced the learning rate without improvement of the validation loss. 

- Investigated its ‘Final validation MAE’ with 200 epochs on GPU (Tesla T4) and predictions, then repeat with 3D data as descriptors and compared their performance.

Please find more details and code in the link:

https://colab.research.google.com/drive/1jH3dWjqh24FpDSyEQ1d2ynV3I2aEvhif?usp=sharing


### Machine learning-aided materials flow

<img width="1048" alt="Screenshot 2023-03-27 at 10 21 10" src="https://user-images.githubusercontent.com/98719524/230827741-3d6a560a-2a70-4216-8efd-59356d3d736b.png">



#### Generative adversarial network (GAN)

Sample generated by simulation tools, which is considered as a generative model for the data as they can be used to generate data of the same complexity and format as the actual experimental data. we can build a simulation modle to generate these data to replace the experiments, which can be used in generative Adversarial network(GAN).

#### Some challenges

There are several open challenges that need to be overcome in order to move to larger systems, longer time scales, higher data efficiency, better generalization and transferability, and eventually more accurate and realistic applications.

#### GNN-for-Materials-Science

1. A Systematic Survey of Chemical Pre-trained Models
https://github.com/junxia97/awesome-pretrain-on-molecules

2. A Gentle Introduction to Graph Neural Networks
https://distill.pub/2021/gnn-intro/

3. Understanding Convolutions on Graphs
https://distill.pub/2021/understanding-gnns/

4. Graph neural networks for materials science and chemistry
https://www.nature.com/articles/s43246-022-00315-6

5. Graph-based deep learning frameworks for molecules and solid-state materials
https://www.sciencedirect.com/science/article/abs/pii/S0927025621000574

6. TensorFlow Graph Neural Network Samples
https://github.com/microsoft/tf-gnn-samples/

7. Semi-Supervised Classification with Graph Convolutional Networks
https://arxiv.org/abs/1609.02907

8. Neural Message Passing for Quantum Chemistry
https://arxiv.org/abs/1704.01212

9. Graph Attention Networks(one of the first model to use attention as a graph convolutional operation)
https://arxiv.org/abs/1710.10903

10. Simplifying Graph Convolutional Networks
https://arxiv.org/pdf/1902.07153.pdf

11. Temporal Graph Networks for Deep Learning on Dynamic Graphs
https://arxiv.org/abs/2006.10637

12. Representation Learning for Dynamic Graphs: A Survey
https://arxiv.org/abs/1905.11485


**Important Names in GNNs**

**Jure Leskovec** (Stanford, chief scientist @ Pinterest) 

**William Hamilton** (McGill, wrote the book on GNNs)

**Key Readings**

1. Graph Representation Learning Textbook (Hamilton, 2020)
(Especially Chapters 5-7 to catch up to the cutting-edge and you can actually do some research based on that)

2. A Comprehensive Survey on Graph Neural Networks (2019)

3. Hamilton's class on GNNs from Winter 2020

https://cs.mcgill.ca/~wIh/comp766/schedule.html

For some anecdotes and stories about GNN, search on Youtube from the people who created these algorithms whenever possible. Talks can provide context that papers cannot.
