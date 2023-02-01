# CNN-for-Molecular-Properties-
Trained a convolutional neural network (CNN) for image analysis and pattern recognition with molecular dataset QM9 and toolbox SchNetPack on Google Colab.

Chose atomization energy as the imput and prediction parameter, and 1000 training examples in 2D/3D dataset for validation, successfully reduced the learning rate without improvement of the validation loss. 

Investigated its ‘Final validation MAE’ with 200 epochs on GPU (Tesla T4) and predictions, then repeat with 3D data as descriptors and compared their performance.

