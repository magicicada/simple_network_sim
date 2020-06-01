# simple_network_sim

> :warning: **The code and data in this repository are currently for testing and development purposes only.**

[![Build Status](https://travis-ci.org/ScottishCovidResponse/simple_network_sim.svg?branch=master)](https://travis-ci.org/ScottishCovidResponse/simple_network_sim)
[![Code Coverage](https://codecov.io/github/ScottishCovidResponse/simple_network_sim/coverage.svg?branch=master&token=)](https://codecov.io/gh/ScottishCovidResponse/simple_network_sim)

## Summary

Adaptation of a simple network simulation model to COVID-19 (forked from https://github.com/magicicada/simple_network_sim to be brought into the SCRC consortium GitHub organisation - this, the SCRC owned repository is the main repository for development). Similar models have previously been used to model other disease outbreaks with different characteristics to COVID-19.

## Features

simple_network_sim represents a geographical area (e.g. Scotland) as a series of connected nodes in a network. These could be counties, health board areas, hospitals or even, in a special case, individuals. Each node is of the same type in a given network (e.g. all counties). Some nodes have more movement between them than others.

!["Network"](assets/network.png)

**Network representing a geographical area, thicker lines indicate more movement**

Within each node is a population, stratified by age group into **y**oung, **m**ature and **o**ld. 

The progress of the epidemic is modelled within nodes using compartments describing the number of people in various disease states within the node. There is one of these sets of compartments per node.

!["Compartments"](assets/colourfulCompartments.png)

**Disease state compartments within each network node**

As simulated time incrementally moves forward, the model predicts the number of people in each disease state in each node taking into account:

- Movement of people between nodes
- Progression through disease state compartments within each node (affected by mixing between age stratified sub-populations)

A more detailed model overview [here](model_overview_simple_network_sim.md).

## Contributing

Please read our [contributing information](contributing.md).

## Installation

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (it also works with [anaconda](https://docs.anaconda.com/anaconda/install/), but we do not need the extra packages). With conda installed, run the following commands to create the virtual environment and activate it:

```
conda env create -f environment.yml
conda activate simple_network_sim
```

## Reproducible builds

In order to ensure reproducibility, we export a [spec-file.txt](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments) whenever we change dependencies for this project. The file pins the version for every dependency and subdependencies of the project.  Since the CI use that file to set up the environment, it is linux specific. A conda environment can be created with it by running:

```
conda create --name simple_network_sim --file spec-file.txt
conda activate simple_network_sim
```

The file can be created with the following command:

```
conda list --explicit > spec-file.txt
```

## Tests

After activating your conda environment, execute the following command:

```{shell}
pytest --cov=simple_network_sim tests
```

## Usage

To run a example case, enter the following at the command prompt:

```{shell}
python -m simple_network_sim.sampleUseOfModel random afilename.pdf
```

that will pick random nodes to infect at each iteration. If you prefer to pass a seed file, you can do the following:

```{shell}
python -m simple_network_sim.sampleUseOfModel seeded sample_input_files/initial_infection.csv afilename.pdf
```

Use the help command to see a description of all the parameters

```{shell}
pyhon -m simple_network_sim.sampleUseOfModel -h
usage: sampleUseOfModel.py [-h]
                           [--compartment-transition COMPARTMENT_TRANSITION]
                           [--population POPULATION] [--commutes COMMUTES]
                           [--mixing-matrix MIXING_MATRIX] [--time TIME]
                           {random,seeded} ...

Uses the deterministic network of populations model to simulation the disease
progression

positional arguments:
  {random,seeded}
    random              Randomly pick regions to infect
    seeded              Use a seed file with infected regions

optional arguments:
  -h, --help            show this help message and exit
  --compartment-transition COMPARTMENT_TRANSITION
                        Epidemiological rate parameters for movement within
                        the compartmental model. (default: sample_input_files/compartmentTransitionByAge.csv)
  --population POPULATION
                        This file contains age-and-sex-stratified population
                        numbers by geographic unit. (default: sample_input_files/sample_hb2019_pop_est_2018_row_based.csv)
  --commutes COMMUTES   This contains origin-destination flow data during
                        peacetime for health boards (default: sample_input_files/sample_scotHB_commute_moves_wu01.csv)
  --mixing-matrix MIXING_MATRIX
                        This is a sample square matrix of mixing - each column
                        and row header is an age category. (default: sample_input_files/simplified_age_infection_matrix.csv)
  --movement-multipliers MOVEMENT_MULTIPLIERS
                        By using this parameter you can adjust dampening or
                        heightening people movement through time (default: None)
  --time TIME           The number of time steps to take for each simulation
                        (default: 200)
```

Each command has its own set of specific parameters

```{shell}
python -m simple_network_sim.sampleUseOfModel seeded -h
usage: sampleUseOfModel.py seeded [-h] [--trials TRIALS] input output

positional arguments:
  input            File name with the seed region seeds
  output           Name of the PDF file that will be created with the
                   visualisation

optional arguments:
  -h, --help       show this help message and exit
  --trials TRIALS  Number of experiments to run (default: 1)
```
```{shell}
usage: sampleUseOfModel.py random [-h] [--regions REGIONS]
                                  [--age-groups AGE_GROUPS [AGE_GROUPS ...]]
                                  [--trials TRIALS] [--infected INFECTED]
                                  output

positional arguments:
  output                Name of the PDF file that will be created with the
                        visualisation

optional arguments:
  -h, --help            show this help message and exit
  --regions REGIONS     Number of regions to infect (default: 1)
  --age-groups AGE_GROUPS [AGE_GROUPS ...]
                        Age groups to infect (default: ['[17,70)'])
  --trials TRIALS       Number of experiments to run (default: 100)
  --infected INFECTED   Number of infected people in each region/age group
                        (default: 100)
```
 
Descriptions of the data files used can be found in the [data dictionary](sample_input_files/data_dictionary.md).

## Continuous integration

[Continuous integration uses Travis and Codecov](ci.md).

## License

[The 2-Clause BSD License](LICENSE.txt).
