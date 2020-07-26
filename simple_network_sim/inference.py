"""
main module used for running the inference on simple network sim
"""
from __future__ import annotations

import logging.config
import sys
import time
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Type, ClassVar

import numpy as np
import pandas as pd
import scipy.stats as stats
from data_pipeline_api import standard_api

from simple_network_sim import loaders
from simple_network_sim import network_of_populations as ss
from simple_network_sim import sampleUseOfModel as sm

sys.path.append('..')

logger = logging.getLogger(__name__)


class InferredVariable(ABC):
    """
    Abstract class representing a variable to infer in ABC-SMC. To be inferred,
    we require from a parameter to:
    - Sample and retrieve pdf from prior
    - Perturb the parameter and get the perturbation pdf
    - Validate if the parameter is correct
    - Convert to frame to be initialize and run the model
    """
    value: pd.DataFrame

    @staticmethod
    @abstractmethod
    def generate_from_prior(fitter: ABCSMC) -> InferredVariable:
        """ Abstract method for generating a parameter from the prior """

    @abstractmethod
    def generate_perturbated(self) -> InferredVariable:
        """ Abstract method for generating a perturbated copy """

    @abstractmethod
    def validate(self) -> bool:
        """ Abstract method for validating the parameter correctness """

    @abstractmethod
    def prior_pdf(self) -> float:
        """ Abstract method for generating the prior pdf """

    @abstractmethod
    def perturbation_pdf(self, x: pd.DataFrame) -> float:
        """ Abstract method for generating the perturbation pdf """


class InferredInfectionProbability(InferredVariable):
    """
    Class representing inferred infection probability to be used inside ABC-SMC fitter.
    Infection probability is the odd that a contact between an infectious and susceptible
    person leads to a new infection.
    """

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            value: pd.DataFrame,
            mean: float,
            shape: float,
            kernel_sigma: float,
            random_state: np.random.Generator
    ):
        self.value = value
        self.mean = mean
        self.shape = shape
        self.kernel_sigma = kernel_sigma
        self.random_state = random_state

    @staticmethod
    def generate_from_prior(fitter: ABCSMC) -> InferredInfectionProbability:
        """ Sample from prior distribution. For infection probability, we use
        the value in the data pipeline as mean of our prior, and shape parameter
        is fixed at 2. This defines a Beta distribution centered around our prior.

        :param fitter: ABC-SMC fitter object
        :return: New InferredInitialInfections randomly sampled from prior
        """
        shape = fitter.infection_probability_shape
        kernel_sigma = fitter.infection_probability_kernel_sigma
        mean = fitter.infection_probability.at[0, "Value"]
        value = fitter.infection_probability.copy().assign(
            Value=lambda x: fitter.random_state.beta(shape, shape * (1 - x.Value) / x.Value))
        return InferredInfectionProbability(value, mean, shape, kernel_sigma, fitter.random_state)

    def generate_perturbated(self) -> InferredInfectionProbability:
        """ From current parameter, add a perturbation to infection probability and return
        a newly created perturbated parameter:

        .. math::
            P_t^* \sim K(P_t | P_{t-1}) \sim P_{t-1} + \sigma * Uniform([-1,1])

        :return: New parameter which is similar to self up to a perturbation
        """
        value = self.value.copy()
        value.Value = value.Value.apply(lambda x: x + self.kernel_sigma * self.random_state.uniform(-1., 1.))
        return InferredInfectionProbability(value, self.mean, self.shape, self.kernel_sigma, self.random_state)

    def validate(self) -> bool:
        """ Checks that the particle is valid, i.e. that infection probability is
        between 0 and 1.

        :return: Whether the parameter is valid
        """
        return np.all(self.value.Value > 0.) and np.all(self.value.Value < 1.)

    def prior_pdf(self) -> float:
        """ Compute pdf of the prior distribution evaluated at the parameter x.
        Infection probability has a prior a beta distribution. The pdf is evaluated
        at the current value of the parameter.

        :return: pdf value of prior distribution evaluated at x
        """
        return stats.beta.pdf(self.value.at[0, "Value"], self.shape, self.shape * (1 - self.mean) / self.mean)

    def perturbation_pdf(self, x: pd.DataFrame) -> float:
        """ Compute pdf of the perturbation evaluated at the parameter x,
        from the current parameter. In ABC-SMC when a particle is sampled
        from the previous population it is slightly perturbed:

        .. math::
            P_t^* ~ K(P_t | P_{t-1}) ~ P_{t-1} + Uniform([-1,1]) * kernel_sigma

        :param x: Particle to evaluate the pdf at
        :return: pdf value of perturbation from previous particle evaluated at x
        """
        return stats.uniform.pdf(x.at[0, "Value"], self.value.at[0, "Value"] - self.kernel_sigma, 2 * self.kernel_sigma)


class InferredInitialInfections(InferredVariable):
    """
    Class representing inferred initial infections to be used inside ABC-SMC fitter.
    Initial infections are the number of exposed individuals per node at the start
    date of the model.
    """

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            value: pd.DataFrame,
            mean: pd.DataFrame,
            stddev: float,
            stddev_min: float,
            kernel_sigma: float,
            random_state: np.random.Generator
    ):
        self.value = value
        self.mean = mean
        self.stddev = stddev
        self.stddev_min = stddev_min
        self.random_state = random_state
        self.kernel_sigma = kernel_sigma

    @staticmethod
    def _rvs_lognormal(mean: float, stddev: float, stddev_min: float):
        """ Constructs Scipy lognormal object to match a given mean and std
        dev passed as input. The parameters to input in the model are inverted
        from the formulas:

        .. math::
                if X~LogNormal(mu, scale)
            then:
                E[X] = exp{mu + sigma^2 * 0.5}
                Var[X] = (exp{sigma^2} - 1) * exp{2 * mu + sigma^2}

        The stddev is taken as a % of the mean, floored at 10. This
        allows natural scaling with the size of the population inside the
        nodes, always allowing for a minimal uncertainty.

        :param mean: Mean to match
        :param stddev: Std dev to match
        :return: Distribution object representing a lognormal distribution with
        the given mean and std dev
        """
        var = np.maximum((mean * stddev)**2, stddev_min)
        sigma = np.sqrt(np.log(1 + (var / mean**2)))
        mu = np.log(mean / np.sqrt(1 + (var / mean**2)))
        return stats.lognorm(s=sigma, loc=0., scale=np.exp(mu))

    @staticmethod
    def generate_from_prior(fitter: ABCSMC) -> InferredInitialInfections:
        """ Sample from prior distribution. For initial infections, we use
        the values in the data pipeline as mean of our priors, and the std dev
        is taken as a percentage of the mean. This allow the prior to scale
        the uncertainty with the scale of the prior itself.

        :param fitter: ABC-SMC fitter object
        :return: New InferredInitialInfections randomly sampled from prior
        """
        stddev = fitter.initial_infections_stddev
        stddev_min = fitter.initial_infections_stddev_min
        kernel_sigma = fitter.initial_infections_kernel_sigma
        value = fitter.initial_infections.copy().assign(Infected=lambda x: InferredInitialInfections
                                                        ._rvs_lognormal(x.Infected, stddev, stddev_min)
                                                        .rvs(random_state=fitter.random_state))
        return InferredInitialInfections(value, fitter.initial_infections, stddev, stddev_min, kernel_sigma,
                                         fitter.random_state)

    def generate_perturbated(self) -> InferredInitialInfections:
        """ From current parameter, add a perturbation to every parameter and return
        a newly created perturbated particle. For initial infections, we apply a uniform
        perturbation from the current value:

        .. math::
            P_t^* \sim K(P_t | P_{t-1}) \sim P_{t-1} + \sigma * Uniform([-1,1])

        :return: New parameter which is similar to self up to a perturbation
        """
        value = self.value.copy()
        value.Infected = value.Infected.apply(lambda x: x + self.kernel_sigma * self.random_state.uniform(-1., 1.))
        return InferredInitialInfections(value, self.mean, self.stddev, self.stddev_min, self.kernel_sigma,
                                         self.random_state)

    def validate(self) -> bool:
        """ Checks that the particle is valid, i.e. that all initial infections are
        positive or 0.

        :return: Whether the particle is valid
        """
        return np.all(self.value.Infected >= 0.)

    def prior_pdf(self) -> float:
        """ Compute pdf of the prior distribution evaluated at the parameter x.
        As all priors are independent the joint pdf is the product of individual pdfs.
        The pdf is evaluated at the current value of the parameter.

        :return: pdf value of prior distribution evaluated at x
        """
        pdf = 1.
        for index, row in self.mean.iterrows():
            pdf *= InferredInitialInfections._rvs_lognormal(row.Infected, self.stddev, self.stddev_min)\
                                            .pdf(self.value.at[index, "Infected"])

        return pdf

    def perturbation_pdf(self, x: pd.DataFrame) -> float:
        """ Compute pdf of the perturbation evaluated at the parameter ``x``,
        from the current parameter. In ABC-SMC when a particle is sampled
        from the previous population it is slightly perturbed:

        .. math::
            P_t^* \sim K(P_t | P_{t-1}) \sim P_{t-1} + \sigma * Uniform([-1,1])

        As all perturbations are independent the joint pdf is the product
        of individual pdfs.

        :param x: Particle to evaluate the pdf at
        :return: pdf value of perturbation from previous particle evaluated at x
        """
        pdf = 1.
        for index, row in self.value.iterrows():
            pdf *= stats.uniform.pdf(x.at[index, "Infected"], row.Infected - self.kernel_sigma, 2 * self.kernel_sigma)

        return pdf


class Particle:
    """
    Class representing a particle, a collection of parameters that will be
    sampled and eventually inferred. To be used within ABC-SMC, a particle
    must satisfy the following requirements:
    1) We can sample from the priors and get the prior pdf
    2) We should be able to generate a perturbated particle
    """

    inferred_variables_classes: ClassVar[Dict[str, Type[InferredVariable]]] = {
        "infection-probability": InferredInfectionProbability,
        "initial-infections": InferredInitialInfections
    }

    def __init__(self, inferred_variables: Dict[str, InferredVariable]):
        self.inferred_variables = inferred_variables

    @staticmethod
    def generate_from_priors(fitter: ABCSMC) -> Particle:
        """ Generate a particle from the prior distribution. All parameters in the
        particle are independent and therefore we simply random select each parameter
        from its prior distribution. The fitter object provides the random state used
        for random variable generation but also the prior parameters, which are
        the values in the data pipeline.

        :return: New particle generated from prior
        """
        return Particle({variable_name: variable_class.generate_from_prior(fitter)
                         for variable_name, variable_class in Particle.inferred_variables_classes.items()})

    def generate_perturbated(self) -> Particle:
        """ From current particle, add a perturbation to every parameter and return
        a newly created perturbated particle.

        :return: New particle which is similar to self up to a perturbation
        """
        perturbed_variables = {}
        for name, inferred_variable in self.inferred_variables.items():
            perturbed_variables[name] = inferred_variable.generate_perturbated()

        return Particle(perturbed_variables)

    def validate_particle(self) -> bool:
        """ Checks that the particle is valid, simply checks that all parameters inside
        respect their supports, i.e. the values are attainable. It may return False for
        example if the uniform perturbation pushed a lognormally-distribution value
        below 0.

        :return: Whether the particle is valid
        """
        return all(variable.validate() for variable in self.inferred_variables.values())

    @staticmethod
    def resample_and_perturbate(
            particles: List[Particle],
            weights: List[float],
            random_state: np.random.Generator
    ) -> Particle:
        """ Resampling part of ABC-SMC, selects randomly a new particle from the
        list of previously accepted. Then perturb it slightly. This causes
        the persistence and selection of fit particles in a manner similar to
        evolution algorithms. The perturbation is done until the candidate is valid,
        this is because the perturbation is unaware of the support of the distribution
        of the particle.

        :param particles: List of particles from previous population
        :param weights: List of weights of particles from previous population
        :param random_state: Random state for random sampling into particles
        :return: Newly created particles sampled from previous population and perturbated
        """
        while True:
            particle = random_state.choice(particles, p=weights / np.sum(weights))
            particle = particle.generate_perturbated()

            if particle.validate_particle():
                return particle

    def prior_pdf(self) -> float:
        """ Compute pdf of the prior distribution evaluated at the particle x.
        As all priors are independent the joint pdf is the product
        of individual pdfs. The pdf is evaluated at the current particle.

        :return: pdf value of prior distribution evaluated at x
        """
        pdf = 1.
        for key in self.inferred_variables:
            pdf *= self.inferred_variables[key].prior_pdf()

        return pdf

    def perturbation_pdf(self, x: Particle) -> float:
        """ Compute pdf of the perturbation evaluated at the particle x,
        from the current particle. In ABC-SMC when a particle is sampled
        from the previous population it is slightly perturbed:

        .. math::
            P_t^* \sim K(P_t | P_{t-1})

        (Usually a uniform perturbation around the previous value).
        As all perturbations are independent the joint pdf is the product
        of individual pdfs.

        :param x: Particle to evaluate the pdf at
        :return: pdf value of perturbation from previous particle evaluated at x
        """
        pdf = 1.
        for key in self.inferred_variables:
            pdf *= self.inferred_variables[key].perturbation_pdf(x.inferred_variables[key].value)

        return pdf


# pylint: disable=too-many-instance-attributes
class ABCSMC:
    """
    Class to wrap inference routines for the ABC SMC inference fitting. This algorithm
    provides a list of samples distribution with the posterior pdf of parameters
    given the data. This class is fairly tightly coupled to the simple network
    sim network, sacrificing abstraction for speed and clarity. We rely on the
    ``Particle`` class in which all the abstraction about parameters reside.

    Algorithm (briefly):
    Set parameters ``smc_iteration``, ``n_particles``, ``threshold``

    For iteration in smc_iterations:
        While accepted_particles < n_particles:
            p = sample randomly chosen particle from accepted at previous iteration
            p = perturb p Using uniform perturbation
            distance = run model with particle p

            if distance < threshold:
                Particle is accepted for the current population
                Compute weight for current particle

    References:
        https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2008.0172
        https://en.wikipedia.org/wiki/Approximate_Bayesian_computation
        https://pyabc.readthedocs.io/en/latest/index.html
    """

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            parameters: pd.DataFrame,
            historical_deaths: pd.DataFrame,
            compartment_transition_table: pd.DataFrame,
            population_table: pd.DataFrame,
            commutes_table: pd.DataFrame,
            mixing_matrix_table: pd.DataFrame,
            infection_probability: pd.DataFrame,
            initial_infections: pd.DataFrame,
            infectious_states: pd.DataFrame,
            trials: pd.DataFrame,
            start_end_date: pd.DataFrame,
            movement_multipliers_table: pd.DataFrame,
            stochastic_mode: pd.DataFrame,
            random_seed: pd.DataFrame
    ):
        self.historical_deaths = loaders.readHistoricalDeaths(historical_deaths)
        parameters = loaders.readABCSMCParameters(parameters)

        self.n_smc_steps = parameters["n_smc_steps"]
        self.n_particles = parameters["n_particles"]
        self.infection_probability_shape = parameters["infection_probability_shape"]
        self.infection_probability_kernel_sigma = parameters["infection_probability_kernel_sigma"]
        self.initial_infections_stddev = parameters["initial_infections_stddev"]
        self.initial_infections_stddev_min = parameters["initial_infections_stddev_min"]
        self.initial_infections_kernel_sigma = parameters["initial_infections_kernel_sigma"]

        assert self.n_smc_steps > 0
        assert self.n_particles > 0
        assert self.infection_probability_shape > 0.
        assert self.infection_probability_kernel_sigma > 0.
        assert self.initial_infections_stddev > 0.
        assert self.initial_infections_stddev_min > 0.
        assert self.initial_infections_kernel_sigma > 0.

        self.compartment_transition_table = compartment_transition_table
        self.population_table = population_table
        self.commutes_table = commutes_table
        self.mixing_matrix_table = mixing_matrix_table
        self.infection_probability = infection_probability
        self.initial_infections = initial_infections
        self.infectious_states = infectious_states
        self.trials = trials
        self.start_end_date = start_end_date
        self.movement_multipliers_table = movement_multipliers_table
        self.stochastic_mode = stochastic_mode
        self.random_seed = random_seed
        self.random_state = np.random.default_rng(loaders.readRandomSeed(random_seed))

        assert trials.at[0, "Value"] == 1

        self.threshold = np.inf
        self.fit_statistics: Dict[int] = {}

    def fit(self) -> Tuple[List[Particle], List[float], List[float]]:
        """ Performs ABC-SMC iterative procedure for finding posterior distributions
        of model parameters given priors and data. In brief, iteratively samples
        particles, keeping at each round only ones with good fitting criteria.

        References:
        https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2008.0172
        https://en.wikipedia.org/wiki/Approximate_Bayesian_computation

        :return: List of particles selected at the end of the procedure, and weights
        associated
        """
        prev_particles: List[Particle] = []
        prev_weights: List[float] = []
        distances: List[float] = []

        for smc_step in range(self.n_smc_steps):
            logger.info("SMC step %d/%d", smc_step + 1, self.n_smc_steps)
            prev_particles, prev_weights, distances = self.sample_particles(smc_step, prev_particles, prev_weights)
            self.update_threshold(distances)

        return prev_particles, prev_weights, distances

    def update_threshold(self, distances: List[float]):
        """ Updates threshold using distances found on previous round.
        The median is used as a criterion for next round.

        :param distances: List of distances found for particles in previous round
        :return: New threshold to use for next round, median of distances from previous round
        """
        self.threshold = np.percentile(distances, 50)

    def sample_particles(
            self,
            smc_step: int,
            prev_particles: List[Particle],
            prev_weights: List[float]
    ) -> Tuple[List[Particle], List[float], List[float]]:
        """ Internal single iteration of ABC-SMC. Sampling particles (from prior or previous accepted ones),
        until enough particles pass a goodness-of-fit threshold.

        :param smc_step: ABC-SMC iteration number
        :param prev_particles: List of particles accepted on previous round
        :param prev_weights: List of particles weights accepted on previous round
        :return: List of particles and weights accepted at current ABC-SMC round
        """
        t0 = time.time()

        particles = []
        weights = []
        distances = []

        particles_accepted = 0
        particles_simulated = 0
        while particles_accepted < self.n_particles:

            if smc_step == 0:
                particle = Particle.generate_from_priors(self)
            else:
                particle = Particle.resample_and_perturbate(prev_particles, prev_weights, self.random_state)

            result = self.run_model(particle)
            distance = self.compute_distance(result)

            if distance <= self.threshold:
                logger.info("Particle accepted with distance %d", distance)
                weight = ABCSMC.compute_weight(smc_step, prev_particles, prev_weights, particle)
                particles.append(particle)
                weights.append(weight)
                distances.append(distance)
                particles_accepted += 1

            particles_simulated += 1

        logger.info("Particles accepted %d/%d", particles_accepted, particles_simulated)
        self.add_iteration_statistics(smc_step, particles, weights, particles_simulated, distances, t0)

        return particles, weights, distances

    def run_model(self, particle: Particle) -> pd.DataFrame:
        """ Run models using current particle as parameters.

        :param particle: Particle under consideration
        :return: Model run results for current particle
        """
        network = ss.createNetworkOfPopulation(
            self.compartment_transition_table,
            self.population_table,
            self.commutes_table,
            self.mixing_matrix_table,
            self.infectious_states,
            particle.inferred_variables["infection-probability"].value,
            particle.inferred_variables["initial-infections"].value,
            self.trials,
            self.start_end_date,
            self.movement_multipliers_table,
            self.stochastic_mode,
            self.random_seed
        )
        results = sm.runSimulation(network)
        aggregated = sm.aggregateResults(results)
        return aggregated.output

    def compute_distance(self, result: pd.DataFrame) -> float:
        """ Computes distance between target and model run with current particle.
        For dynamical systems such as epidemiological models, the distance generally
        used is the root mean squared distance between model and historical outputs.
        e.g. for a dynamical system which outputs `y(t)` (number of deaths, number
        of infected, etc):

        .. math::
            \sqrt{\sum_{t=1,...,T} ( y_{model}(t) - y_{reality}(t) )^2}

        In our case, `y(t)` is the number of deaths per node and per week.

        :param result: Model run results
        :return: distance value between model run and target
        """
        result_by_node = (
            result
            .query("state == 'D'")
            .groupby(["date", "node"])
            .sum()
            .reset_index()
            .assign(date=lambda x: pd.to_datetime(x.date))
            .pivot(index="date", columns="node", values="total")
            .diff()
            .resample('7D').sum()
        )

        distance = (result_by_node - self.historical_deaths)**2
        distance = np.sqrt(distance.sum().sum() / distance.count().sum())
        return distance

    @staticmethod
    def compute_weight(
            smc_step: int,
            particles: List[Particle],
            weights: List[float],
            particle: Particle
    ) -> float:
        """ Compute weights of particle as per the ABC-SMC algorithm.
        As per the reference article [#tonistumpf]_, the weights are
        updated as per the formula:

        .. math::
            w_t^i = \frac{\PI(\Theta_t^i)}{\sum_{j=1}^N w_{t-1}^j K(\Theta_t^{j-1}, \Theta_t^{j})}

        .. [#tonistumpf] Toni, Tina, and Michael P. H. Stumpf.
                      “Simulation-Based Model Selection for Dynamical
                      Systems in Systems and Population Biology”.
                      Bioinformatics 26, no. 1, 104–10, 2010.
                      doi:10.1093/bioinformatics/btp619.

        :param smc_step: Step number of ABC-SMC algorithm
        :param particles: List of accepted particles in the previous run
        :param weights: List of weights of accepted particles in the previous run
        :param particle: Particle under consideration
        :return: Weight of the particle under consideration
        """
        if smc_step == 0:
            return 1.

        num = particle.prior_pdf()
        denom = sum(weights[i] * p.perturbation_pdf(particle) for i, p in enumerate(particles))

        return num / denom

    # pylint: disable=too-many-arguments
    def add_iteration_statistics(
            self,
            smc_step: int,
            particles: List[Particle],
            weights: List[float],
            particles_simulated: int,
            distances: List[float],
            t0: float
    ):
        """ Log statistics of each iteration of ABC-SMC algorithm

        :param smc_step: Step number of ABC-SMC algorithm
        :param particles: Accepted particles
        :param weights: Weights of accepted particles
        :param particles_simulated: Number of accepted particles
        :param distances: List of distances generated by accepted particles
        :param t0: Time at start of iteration
        """
        self.fit_statistics.setdefault(smc_step, {})
        self.fit_statistics[smc_step]["particles"] = particles
        self.fit_statistics[smc_step]["weights"] = weights
        self.fit_statistics[smc_step]["particles_accepted"] = len(particles)
        self.fit_statistics[smc_step]["particles_simulated"] = particles_simulated
        self.fit_statistics[smc_step]["distances"] = distances
        self.fit_statistics[smc_step]["threshold"] = self.threshold
        self.fit_statistics[smc_step]["time"] = f"{time.time() - t0:.0f}s"

    def summarize(self, particles: List[Particle], weights: List[float], distances: List[float], t0: float) -> Dict:
        """ Summarize ABC-SMC run, by assembling all fit statistics
        and final list of particles into a dictionary.

        :param particles: Accepted particles
        :param weights: Weights of accepted particles
        :param distances: Distances of accepted particles
        :param t0: Time just before fit started
        """
        results = {
            "fit_statistics": self.fit_statistics,
            "particles": particles,
            "weights": weights,
            "distances": distances,
            "best_particle": particles[int(np.argmin(distances))],
            "time": time.time() - t0
        }

        return results


def run_inference(config, uri: str = "", git_sha: str = "") -> Dict:
    """Run inference routine

    :param config: Config file name
    :type config: string
    :param uri: Git uri used
    :param git_sha: git_sha used
    :return: Result runs for inference
    """

    with standard_api.StandardAPI(config, uri=uri, git_sha=git_sha) as store:
        abcsmc = ABCSMC(
            store.read_table("human/abcsmc-parameters", "abcsmc-parameters"),
            store.read_table("human/historical-deaths", "historical-deaths"),
            store.read_table("human/compartment-transition", "compartment-transition"),
            store.read_table("human/population", "population"),
            store.read_table("human/commutes", "commutes"),
            store.read_table("human/mixing-matrix", "mixing-matrix"),
            store.read_table("human/infection-probability", "infection-probability"),
            store.read_table("human/initial-infections", "initial-infections"),
            store.read_table("human/infectious-compartments", "infectious-compartments"),
            store.read_table("human/trials", "trials"),
            store.read_table("human/start-end-date", "start-end-date"),
            store.read_table("human/movement-multipliers", "movement-multipliers"),
            store.read_table("human/stochastic-mode", "stochastic-mode"),
            store.read_table("human/random-seed", "random-seed"),
        )

        t0 = time.time()
        particles, weights, distances = abcsmc.fit()
        summary = abcsmc.summarize(particles, weights, distances, t0)

    return summary


def main(argv):
    args = sm.build_args(argv)
    sm.setup_logger(args)
    logger.info("Running inference ABC SMC...")

    t0 = time.time()
    run_inference("../config_inference.yaml")

    logger.info("Writing output")
    logger.info("Took %.2fs to run the inference.", time.time() - t0)


if __name__ == "__main__":
    logger = logging.getLogger(f"{__package__}.{__name__}")
    main(sys.argv[1:])
