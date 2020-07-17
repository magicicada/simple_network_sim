# Extractors

The scripts in this folder will extract the data from the best datasource we can find in order to have a realist run of
the simple network sim model. They are not necessarily downloading them from the official SCRCdata resource, although
they might be partially doing that. The resulting files from the scripts in here can be plugged into the model by
copying them to the appropriate folder inside `sample_input_files`.

We expect the scripts in here will, at some point, make it into the SCRCdata repo as post-processing scripts. However,
as the mechanism are not yet in place in SCRCdata or SCRCdataAPI to have these scripts fully integrated, we decided to
have the basic logic needed implemented in here. They can be later adapted to read and write files as needed.

Each (non-test) `.py` file is a self-contained implementation that will generate an input for the simple_network_sim
model. Documentation on how to use them and which files to use are in the docstrings of each model in here.

## Environment

All scripts in this folder are written in Python and an environment to run them can be created using conda. Use the
`environment.yml` file within this directory to create a conda environment with all the needed dependencies:

```bash
conda env create -f environment.yml extractors
conda activate extractors
```

You will need to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) before running the commands above.
