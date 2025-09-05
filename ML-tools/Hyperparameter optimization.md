# Hyperparameter optimization

When training ML models, performance often depends on a large number of hyperparameters. For instance, in a fully connected deep neural network, important hyperparameters include the number of hidden layers, the number of nodes per layer, the choice of activation function, learning rate, and more. Manually tuning these parameters in high-dimensional spaces is typically non-intuitive and time-consuming, making automated hyperparameter optimization methods particularly useful.

## 1. What are hyperparameters?

Hyperparameters are settings you choose before training a model, unlike model parameters (weights) which are learned from data.

Examples in a neural network:
- Number of layers
- Number of neurons per layer
- Learning rate
- Activation function (ReLU, Sigmoid, etc.)

Choosing the right hyperparameters strongly affects model performance, but figuring them out manually is tricky, especially when there are many.

## 2. Why automated hyperparameter optimization?

- Manually testing each combination is slow and inefficient.
- Automated methods can explore the “hyperparameter space” intelligently to find combinations that give better performance, which is especially important in high-dimensional spaces (many hyperparameters).


## 3. How wandb helps

Weights & Biases (wandb) is a tool that simplifies hyperparameter optimization. It automates the trial-and-error of hyperparameter tuning, tracks all experiments, and helps you make sense of which settings work best.

It provides a feature called “sweeps”, which lets you:

- Define which hyperparameters to try and their ranges.
- Choose a search strategy, like:
   - Random search: Try random combinations.
   - Grid search: Try all combinations in a grid (less efficient for many hyperparameters).
   - Bayesian optimization: Uses previous results to make smarter choices about what to try next.
- Track performance of each combination automatically.
- wandb also gives visualizations, so you can quickly see which hyperparameters worked best and compare experiments.
- It integrates easily with your Python/ML code. You don’t need to write a lot of extra code to log metrics.
- It balances ease of use and power, making it ideal for optimizing GCNNs, CGCNNs, or other models.



**In our group, we often use [Weights & Biases (wandb)](https://wandb.ai/) for hyperparameter optimization.** Wandb provides a flexible and user-friendly interface for running sweeps, which support a range of optimization strategies such as Bayesian optimization, random search, and grid search. It also integrates seamlessly with existing ML workflows and provides powerful visualization tools to analyze results. Depending on the task, other tools may also be appropriate, but wandb strikes a good balance between ease of use and capability.



## Usage

Wandb offers comprehensive features for logging training metrics such as loss and accuracy, tracking hyperparameters, organizing and comparing different training runs, and automating hyperparameter optimization through sweep configurations. wandb also allows users to save and version models, datasets, and other files as "artifacts," making it a valuable tool for reproducibility and collaborative work. If you're new to wandb, you can think of it as a fitness tracker for ML—it monitors every detail of your experiments so you can analyze and improve them more effectively.

Wandb is free for academic use. **If you plan to use it, please ask Jiayu to add you to our academic group workspace on the platform.** This ensures all your experiments are organized under the group account and benefit from team features. **Check out their [official documentation](https://docs.wandb.ai/), including the [quickstart guide](https://docs.wandb.ai/quickstart/), [API reference](https://docs.wandb.ai/ref/), and a collection of [example projects](https://github.com/wandb/examples) on GitHub.com.**

## Example codes

### Installation

To get started, you need to install wandb using either Conda or `pip`. In your terminal, simply run:
```
conda install -c conda-forge wandb
```
or
```
pip install wandb
```

### Logging in

Once installed, you must run through the command line:
```
wandb login
```
or programmatically in Python using:
```
import wandb
wandb.login()
```
to log in to your wandb account on each machine you plan to use.

### Initializing a run

To begin tracking an experiment, initialize a run at the start of your script using the `wandb.init()` function, where you can specify your project name, team or entity name, and optionally log hyperparameters such as the learning rate, number of epochs, and batch size:
```
import wandb

wandb.init(
    project="my-ml-project",   # Name of your project
    entity="my-team",          # (Optional) Team/account name
    config={                   # (Optional) Hyperparameters
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32
    }
)
```
This initialization step creates a new run in your wandb dashboard, where all logs and metadata from that execution will be stored.

### Logging data

During model training, you can log metrics such as loss, accuracy, or any other custom value using the `wandb.log()` function inside your training loop:
```
for epoch in range(epochs):
    # ... training code ...
    wandb.log({"epoch": epoch, "loss": loss, "accuracy": acc})
```
Wandb also allows you to log or update hyperparameters at any time using `wandb.config.update()`:
```
wandb.config.update({"dropout": 0.5})
```
In addition to scalar metrics, wandb supports logging media such as images, plots, or custom visualizations, which is especially useful for tasks like computer vision or debugging:
```
wandb.log({"sample_image": [wandb.Image(img_array, caption="Sample")]})
```

### Hyperparameter sweeps

One of the most powerful features of wandb is its support for automated hyperparameter optimization, known as sweeps. To run a sweep, you first define a sweep configuration, which specifies the optimization method (such as Bayesian, random, or grid search), the objective metric to optimize (e.g., validation accuracy), and the range or list of values for each hyperparameter. This configuration can be written as a Python dictionary or a YAML file:
```
sweep_config = {
    "method": "bayes",  # or "grid", "random"
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "batch_size": {"values": [16, 32, 64]},
        "optimizer": {"values": ["adam", "sgd"]}
    }
}
```
After defining the sweep, you register it using `wandb.sweep()` and then launch one or more agents using `wandb.agent()` to execute training runs with different parameter combinations:
```
sweep_id = wandb.sweep(sweep_config, project="my-ml-project")

def train():
    wandb.init()
    config = wandb.config
    # ... use config.learning_rate, config.batch_size, etc. ...
    # ... training code ...
    wandb.log({"val_accuracy": val_acc})

wandb.agent(sweep_id, function=train, count=10)
```
If you prefer, you can also launch sweeps from the command line using the wandb sweep and wandb agent commands:
```
wandb sweep sweep_config.yaml
wandb agent <entity/project/sweep_id>
```

### Saving and managing artifacts

Beyond metrics and hyperparameters, wandb lets you track versioned files using artifacts. Artifacts are useful for saving trained models, processed datasets, or any other file you want to associate with a specific experiment. To log an artifact, create a new artifact object, add the desired files to it, and use `wandb.log_artifact()` to upload it:
```
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pt")
wandb.log_artifact(artifact)
```
To retrieve and use an artifact in a later run, use `wandb.use_artifact()` and `artifact.download()` to access the stored files:
```
artifact = wandb.use_artifact("model:latest")
artifact_dir = artifact.download()
```

### Organizing projects and runs

Wandb provides several ways to organize your experiments. Projects serve as logical containers for related runs—such as all experiments from a single paper or task. Each script execution corresponds to a run, which is tracked under a given project. You can further organize runs by assigning them to a group, which is especially useful for keeping all runs from a sweep together. Tags can also be added to runs for filtering and categorization:
```
wandb.init(project="my-ml-project", group="experiment-1", tags=["baseline", "resnet"])
```
These features help you stay organized and make it easier to compare different models and training settings.

### Visualizing results
Once your experiments are running, you can monitor and analyze them through the web dashboard at their [online portal]([wandb.ai](https://wandb.ai/)). The dashboard provides a comprehensive set of tools to compare runs, visualize training curves, track the effects of hyperparameters, and export results for further analysis. You can also generate reports and share them with collaborators.

### Best practices

To use wandb effectively, it's essential to follow a few best practices:
- Always log all relevant hyperparameters and training metrics, and use clear, consistent naming conventions for your projects, runs, and artifacts. Group and tag your runs meaningfully, and organize sweeps systematically to ensure thorough coverage of hyperparameters. Save trained models and crucial files as artifacts, ensuring your experiments are fully reproducible. Add notes and documentation to your runs to make your work understandable in the future, especially when collaborating with others.
- If you encounter issues while using wandb, most can be resolved by checking your login status and verifying your API key.
- In situations where you're working offline, you can initialize runs in offline mode using `wandb.init(mode="offline")` and later sync them with `wandb sync`.
- To debug sweep-related issues, it's recommended to test your training function independently before launching a large batch of runs.
