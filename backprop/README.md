# Prerequisites 🐍

The only prerequisite is to install the Conda package manager. There are two options to do that:

- [The Anaconda distribution](https://www.anaconda.com/products/individual),
  which is a huge full featured distribution with many things included, like a graphical interface.
- [The Miniconda distribution](https://docs.conda.io/en/latest/miniconda.html), which is a leaner distribution, and is command line only.

Miniconda is good if you know your way around the command line. Otherwise I would suggest to choose the Anaconda distribution.

# Setup 🚧

To create your environment from scratch, open a terminal

```
conda env create -f environment.yml
```

Then activate your environment:

```
conda activate rl-olfaction
```

# Time to start working finally 👷

If you made it this far, now you can run JupyterLab 🚀 with the following command:

```
jupyter lab
```

If you later need to add new packages to your environment, just list them in the `environment.yml` file, and then update your environment with:

```
conda env update -f environment.yml
```

# Troubleshooting ⚠️

When you open your notebook you should see at the top right a kernel called `Python 3 (ipykernel)` (note the `ipykernel` between parenthesis). If for some reason you don't see it, you can manually add your new environment (kernel) in Jupyter:

```
python -m ipykernel install --user --name=rl-olfaction
```
