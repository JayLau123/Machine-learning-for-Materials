# Conda and package management

Managing software environments and dependencies is crucial for ensuring reproducibility, consistency, and efficiency in computational research, particularly in ML projects where various libraries and frameworks are constantly evolving.

[Conda](https://en.wikipedia.org/wiki/Conda_(package_manager)) provides a powerful, user-friendly solution for creating **isolated** environments and managing packages across platforms, helping researchers avoid conflicts between dependencies and ensuring that their codes run reliably on different machines. Understanding how to use Conda effectively is a foundational skill for maintaining clean workflows and collaborating within our research group.

`Conda` and `pip` are both package managers for Python, but they differ in scope and functionality. 

Conda is a general-purpose package and environment manager that can:

- Install both Python and non-Python dependencies (such as C/C++ libraries or CUDA), making it ideal for HPC and ML workflows. 
- It also includes built-in environment management, allowing users to create isolated environments with specific Python versions and packages.
-
 
In contrast, `pip` is the [default Python package installer](https://en.wikipedia.org/wiki/Pip_(package_manager)) that pulls packages from the [Python Package Index (PyPI)](https://pypi.org/) and handles only Python dependencies. 

While `pip` is lightweight and widely used, it lacks environment management features and has more limited dependency resolution. In practice, Conda is preferred for complex or cross-language stacks, while `pip` is suitable for **Python-only** setups. **When using both, it's best to install conda packages first to minimize dependency conflicts.**


## Conda v.s. Miniconda

**To get started with Conda, we recommend installing [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main), a free, miniature distribution of Conda that includes only the package manager and Python. This is lighter than [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/main) and allows more control over which packages are installed.** For the general installation of Miniconda, please take a look at their [official tutorial](/docs/getting-started/miniconda/install#macos-linux-installation).


While Anaconda is convenient, it installs hundreds of packages by default, making environments large, slow, and harder to manage—especially in HPC settings. 
Miniconda is a lightweight alternative that provides full control by starting with just Python and Conda, allowing you to install only what you need. This makes it faster, more flexible, and better for reproducible computational research. 

Alternatively, [Mamba](https://github.com/mamba-org/mamba) is a drop-in replacement for Conda, offering much faster dependency resolution thanks to its C++ backend. **For most of our research workflows, Miniconda (or Mamba) is a cleaner, more efficient choice than Anaconda.**

**Conda has already been installed in [our workstations](../Computing%20resources/In-house%20workstations.md) by UB SENS**, which manages the admin (sudo) access. To activate your access to Conda, you need to run:
```
use conda
```
**For [our workstations](../Computing%20resources/In-house%20workstations.md), it is essential to modify the default location where Conda stores downloaded packages and installed environments.** By default, Conda uses your home directory, which has limited storage space. To avoid filling it up, you should redirect package storage to their data drives, which have much more available space. You can do this by running these two commands:
```
conda config --add pkgs_dirs /data/users/[UBITName]/.conda/pkgs
conda config --add envs_dirs /data/users/[UBITName]/.conda/envs
```
This step is crucial for preventing space issues and ensuring stable package management. To reduce disk space, run:
```
conda clean --yes --all
```
to reduce the size of Conda directories by removing unused packages.

## Conda and CCR

Conda is not straightforwardly supported on CCR's HPC systems and should not be directly installed in the home or project directories. For details, refer to [our documentation on UB CCR's HPC clusters](../Computing%20resources/UB%20high-performance%20computing%20(HPC)%20clusters.md). **UB CCR recommends [using their existing Python modules](https://docs.ccr.buffalo.edu/en/latest/software/modules/#python), [building custom module bundles with Easybuild](https://docs.ccr.buffalo.edu/en/latest/howto/easybuild/), or [using containers with Conda](https://docs.ccr.buffalo.edu/en/latest/howto/containerization/)**. [An example container setup](https://github.com/ubccr/ccr-examples/blob/main/containers/2_ApplicationSpecific/conda/README.md) is available in their [CCR examples repository](https://github.com/ubccr/ccr-examples), and you can also check out their [Python documentation](https://docs.ccr.buffalo.edu/en/latest/howto/python/) and the ["Using Python at CCR" course](https://ublearns.buffalo.edu/d2l/le/discovery/view/course/288741) in UB Learns.

## Utilizing Conda

Conda enables the creation of isolated environments with specific Python versions and packages, which is crucial for reproducibility and avoiding dependency conflicts in research projects. 

To get started, you should:

- read [Conda's official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- utilize this [cheat sheet](https://kapeli.com/cheat_sheets/Conda.docset/Contents/Resources/Documents/index).

To view help information for Conda commands, you can use the `--help` flag, which provides a summary of the command's purpose, syntax, available options, and examples. You can also use the shorthand `-h` for the same purpose.

### Managing environments

These commands help you create, duplicate, save, and delete environments.

#### Creating a new environment
 
- To create a new environment called `myenv` with Python 3.10 installed, you can run:
```
conda create -n myenv python=3.10
```
- You can also specify additional packages (`numpy`, `pandas`, etc.) in the same command. Use this when starting a new project and want to isolate its dependencies from others.

- Alternatively, you can create an environment from a [YAML file](https://en.wikipedia.org/wiki/YAML). This command reads the `environment.yml` file (i.e., a list of packages and their versions) and creates a new environment based on that specification. **This is useful when setting up a project from someone else or recreating your own environment on a new machine.**
  
```
conda env create -f environment.yml
```



#### Viewing a list of all environments

To list all Conda environments installed on your system and show the path where each one is stored:
```
conda env list
```
or
```
conda info --envs
```
The currently active environment is marked with an asterisk `*`.

#### Cloning an environment

To make an exact copy of the `myenv` environment into a new one named `myenv_clone`:
```
conda create --name myenv_clone --clone myenv
```
It's useful when you want to test new changes **without affecting the original setup.**

#### Removing an environment

To completely delete the `myenv` environment and all of its files:
```
conda remove --name myenv --all
```
Use this when an environment is no longer needed to **free up disk space.**

#### Exporting an environment

To save all the packages and versions in your current environment to `environment.yml`:
```
conda env export > environment.yml
```
Use this to share your environment with others or to preserve it for **reproducibility**.

### Using environments

These commands help you view, enter, and work inside environments.

#### Activating an environment

**Always activate the correct environment before running your codes !!!**

To switch your shell to use the `myenv` environment:
```
conda activate myenv
```
All Python commands and package installs will now affect only this environment. 

#### Deactivating the current environment

**You should deactivate when switching between environments or before closing your terminal.**

To exit the active Conda environment and return to the base system Python:
```
conda deactivate
```

#### Listing all packages in the current environment

To display all packages and versions currently installed in `myenv`:
```
conda list -n myenv
```
This command helps check if a package is already installed or when troubleshooting environment issues.

#### Installing a package in the current environment

To install a specific package in the specified Conda environment:
```
conda install package_name
```
If the package isn't available in Conda, you can potentially use `pip` instead. 

**Mixing `pip` and `Conda` installs carelessly can create conflicts, particularly when `pip` tries to overwrite binaries or pull in incompatible dependencies.** 

Prior to using `pip` to install any packages, make sure it's installed inside your Conda environment with:
```
conda install pip
```
before running:
```
pip install package_name
```
This ensures that `pip` installs packages within the activated environment, not **globally into the system Python**. 

**Avoid using the system-wide `pip` unless absolutely necessary.** 

You can check whether you are running the environment-specific `pip`:

```
which pip
```
and you should get something like:
```
/data/users/[UBITName]/.conda/envs/myenv/bin/pip
```
You can further check what's installed using:
```
conda list
```
or
```
pip list
```
which helps distinguish which packages came from which source.

#### Updating a package in the current environment

You can also activate the environment first, then run:
```
conda update package_name
```

More directly, to update a specific package in the specified environment to the latest compatible version:
```
conda update --name myenv package_name
```


## Setting up Conda kernels for Jupyter Notebook and VS Code 

**A Conda environment is an isolated workspace that includes a specific Python version and a set of installed packages.**

It's widely used in research and development to manage dependencies for different projects without conflicts. However, to use a Conda environment inside [text editors](Text%20editors.md) like **Jupyter Notebook** or **VS Code**, **you need to turn it into a Conda kernel** — that is, register the environment so that it shows up as an available kernel in Jupyter Notebook or VS Code interfaces.


### VS Code

In [VS Code](Text%20editors.md), your Conda environment may not always appear automatically as a kernel option when using Jupyter Notebooks. While the Python extension typically detects environments on its own, **you can manually specify a Conda environment** if needed. 

To do this, open the Command Palette by pressing `Cmd/Ctrl + Shift + P`, select "Python: Select Interpreter," and look for your Conda environment. 

If your desired environment isn't listed, choose "Enter interpreter path" and then click "Find" to browse to the Python executable within your Conda environment. 

Once selected, the environment will be recognized as a valid Jupyter kernel and will appear in the kernel selector dropdown when working with `.ipynb` files in VS Code. This method ensures that even manually created or externally located Conda environments can be used seamlessly in Jupyter Notebooks within VS Code.

### Jupyter Notebook

To set up a Conda environment as a Jupyter kernel for [Jupyter Notebook](Text%20editors.md), start by activating the environment in your terminal using:
```
conda activate myenv
```
Then, install the `ipykernel` package with:
```
conda install ipykernel
```
This package enables the environment to be used as a Jupyter kernel. Finally, register the environment with the command if your Conda environment does not show up as a Conda kernel in Jupyter Notebook:
```
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```
Here, `--name` is the internal identifier, and `--display-name` is the label that will appear in the Jupyter interface. Once registered, your environment will appear as a **selectable kernel** in Jupyter Notebook or JupyterLab, allowing you to run Jupyter Notebooks using the specific packages and Python version inside that environment.

