# Inference current status and next steps

## Documentation

Detailed documentation about inference is available [here](model_description.pdf).

## Code

Currently the inference code is located in:
- [inference.py](../../simple_network_sim/inference.py) for the main code
- [notebook](../notebooks/Inference%20–%20ABC%20SMC.ipynb) with usage and analysis

The code contains all the routines to run the code from the command line, as well as the core code. It relies on inputs from sample_input_files. The inference can be called from the CLI through the command:

```{shell}
python inference.py
```

Or through the notebook, which allows to analyse the outputs and generate graphs.

## Infrastructure and run potential improvements

- Run at more granular scale. This is a bit unclear how to do given the initial state of the model is inferred, and inferring the initial state for hundreds of IZ is not feasible. It would be doable if setting manually the initial infections and not inferring them
- Parallelize the code. While it’s not too hard to parallelize the inner sampling mechanism of ABC SMC using the python multiprocessing package and generate/reject particles in a parallel fashion, it would be much harder if going to an external pool. At HB model the inference takes 1h30 to run with 5 steps, 100 particles
- Adding new parameters to infer should not be too much of a problem code wise, but increase the dimensionality of the model will degrade the ABC SMC inference further. Currently we infer 18 parameters which is already quite a lot
- Run using stochastic mode so that posterior dispersion is realistic. This is not a problem except for the increased run time. I can do it in my machine
- Improved output generation. Currently graphs are only available from the notebook. We can have a pdf file generated for every inference run. Similarly, inferred parameters should be stored somewhere

## Algorithm potential improvements

I’ve made a review of various improvements over ABC SMC and what is possible to do:
- Adaptive population size according to the mean coefficient of variation error criterion, as detailed in paper below. This strategy tries to respond to the shape of the current posterior approximation by selecting the population size such that the variation of the density estimates matches the target variation. I think this is low priority. https://doi.org/10.1007/978-3-319-67471-1_8
- Change uniform transition kernel to e.g. gaussian one. Uniform is used in the original paper but gaussian kernels are common in particle filters and also proposed in the pyABC python package. This is medium priority, and been mentioned by Ben Swallow as well. EERA uses a uniform kernel. https://arxiv.org/abs/1106.6280 
- Stochastic acceptors. The most typical and simple way is to compute the distance between simulated and observed summary statistics, and accept if this distance is below some epsilon threshold. However, also more complex acceptance criteria are possible, in particular when the distance measure and epsilon criteria develop over time. I think this is low priority as the method was mostly introduced to ABC. However, pyABC has this feature.
https://arxiv.org/abs/0811.3355
- Make the perturbation noise adaptive to the particle dispersion rather than flat uniform with a sigma. E.g. the perturbation could be sigma * (max particle – min particle) like is done in EERA. This is high priority and I will do it myself.
