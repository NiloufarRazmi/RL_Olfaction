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
conda activate {{ environment_name }}
```

Add your new environment (kernel) in Jupyter:

```
python -m ipykernel install --user --name={{ environment_name }}
```

# Time to start working finally 👷

If you made it this far, now you can run JupyterLab 🚀 with the following command:

```
jupyter lab
```

Once in JupyterLab, don't forget to change your kernel to {{ environment_name }}.

If you later need to add new packages to your environment, just list them in the `environment.yml` file, and then update your environment with:

```
conda env update -f environment.yml
```
