from __future__ import annotations

import logging.config
import sys
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
from abc import ABC, abstractmethod

from typing import Tuple, List, Any, Dict, Optional

from simple_network_sim import sampleUseOfModel as sm
from simple_network_sim import network_of_populations as ss
from simple_network_sim import data, loaders

sys.path.append('..')

logger = logging.getLogger(__name__)


class InferredVariable(ABC):
    value: Any

    @staticmethod
    @abstractmethod
    def generate_from_prior(fitter: ABCSMC) -> InferredVariable:
        pass

    @abstractmethod
    def generate_perturbated(self) -> InferredVariable:
        pass

    @abstractmethod
    def validate(self) -> bool:
        pass

    @abstractmethod
    def generate_dataframe(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def prior_pdf(self, value: InferredVariable) -> float:
        pass

    @abstractmethod
    def perturbation_pdf(self, value: InferredVariable) -> float:
        pass


class InferredInfectionProbability(InferredVariable):

    def __init__(self, value: float, mean: float, location: float, random_state: np.random.Generator):
        self.value = value
        self.mean = mean
        self.location = location
        self.kernel_sigma = 0.1
        self.random_state = random_state

    @staticmethod
    def generate_from_prior(fitter: ABCSMC) -> InferredInfectionProbability:
        """ Sample from prior distribution. For infection probability, we use
        the value in the data pipeline as mean of our prior, and location parameter
        is fixed at 2. This defines a Beta distribution centered around our prior.

        :param fitter: ABC-SMC fitter object
        :return: New InferredInitialInfections randomly sampled from prior
        """
        location = 2.
        mean = fitter.infection_probability.at[0, "Value"]
        value = stats.beta.rvs(location, location * (1 - mean) / mean, random_state=fitter.random_state)
        return InferredInfectionProbability(value, mean, location, fitter.random_state)

    def generate_perturbated(self) -> InferredInfectionProbability:
        """ From current parameter, add a perturbation to infection probability and return
        a newly created perturbated parameter:

        P_t* ~ K(P_t | P_{t-1}) ~ P_{t-1} + Uniform([-1,1]) * kernel_sigma

        :return: New parameter which is similar to self up to a perturbation
        """
        perturbated_value = self.value + stats.uniform.rvs(-1., 2., random_state=self.random_state) * self.kernel_sigma
        return InferredInfectionProbability(perturbated_value, self.mean, self.location, self.random_state)

    def validate(self) -> bool:
        """ Checks that the particle is valid, i.e. that infection probability is
        between 0 and 1.

        :return: Whether the parameter is valid
        """
        return 0. < self.value < 1.

    def generate_dataframe(self) -> pd.DataFrame:
        """ Generate dataframe to be used in the model run.

        :return: dataframe in the format ingested by the model declarator
        """
        return pd.DataFrame([0., self.value], columns=[0], index=["Time", "Value"]).T

    def prior_pdf(self, x: InferredInfectionProbability) -> float:
        """ Compute pdf of the prior distribution evaluated at the parameter x.
        Infection probability has a prior a beta distribution.

        :param x: Particle to evaluate the pdf at
        :return: pdf value of prior distribution evaluated at x
        """
        return stats.beta.pdf(x.value, self.location, self.location * (1 - self.mean) / self.mean)

    def perturbation_pdf(self, x: InferredInfectionProbability) -> float:
        """ Compute pdf of the perturbation evaluated at the parameter x,
        from the current parameter. In ABC-SMC when a particle is sampled
        from the previous population it is slightly perturbed:

        P_t* ~ K(P_t | P_{t-1}) ~ P_{t-1} + Uniform([-1,1]) * kernel_sigma

        :param x: Particle to evaluate the pdf at
        :return: pdf value of perturbation from previous particle evaluated at x
        """
        return stats.uniform.pdf(x.value, self.value - self.kernel_sigma, 2 * self.kernel_sigma)


class InferredInitialInfections(InferredVariable):

    def __init__(self, value: pd.DataFrame, mean: pd.DataFrame, stddev: float, random_state: np.random.Generator):
        self.value = value
        self.mean = mean
        self.stddev = stddev
        self.random_state = random_state
        self.kernel_sigma = 5.

    @staticmethod
    def _rvs_lognormal(mean: float, stddev: float):
        """ Constructs scipy lognormal object to match a given mean and std
        dev passed as input. The parameters to input in the model are inverted
        from the formulas:
            if X~LogNormal(mu, scale)
        then:
            E[X] = exp{mu + sigma**2 * 0.5}
            Var[X] = (exp{sigma**2} - 1) * exp{2 * mu + sigma**2}

        :param mean: Mean to match
        :param stddev: Std dev to match
        :return: Distribution object representing a lognormal distribution with
        the given mean and std dev
        """
        mean = max(mean, 1e-5)
        var = max((mean * stddev)**2, 100.)  # Minimal stddev of 10 people for initial infections
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
        stddev = 0.2

        value = fitter.initial_infections.copy()
        value.Infected = value.Infected.apply(lambda x: InferredInitialInfections._rvs_lognormal(x, stddev)
                                              .rvs(random_state=fitter.random_state))
        return InferredInitialInfections(value, fitter.initial_infections, stddev, fitter.random_state)

    def generate_perturbated(self) -> InferredInitialInfections:
        """ From current parameter, add a perturbation to every parameter and return
        a newly created perturbated particle. For initial infections, we apply a uniform
        perturbation from the current value:

        P_t* ~ K(P_t | P_{t-1}) ~ P_{t-1} + Uniform([-1,1]) * kernel_sigma

        :return: New parameter which is similar to self up to a perturbation
        """
        perturbated_value = self.value.copy()
        perturbated_value.Infected = perturbated_value.Infected.apply(lambda x: x + self.kernel_sigma * stats.uniform
                                                                      .rvs(-1., 2., random_state=self.random_state))

        return InferredInitialInfections(perturbated_value, self.mean, self.stddev, self.random_state)

    def validate(self) -> bool:
        """ Checks that the particle is valid, i.e. that all initial infections are
        positive or 0.

        :return: Whether the particle is valid
        """
        return np.all(self.value.Infected >= 0.)

    def generate_dataframe(self) -> pd.DataFrame:
        """ Generate dataframe to be used in the model run. In this case the value
        stored is already in dataframe format and hence no additional work is required

        :return: dataframe in the format ingested by the model declarator
        """
        return self.value

    def prior_pdf(self, x: InferredInitialInfections) -> float:
        """ Compute pdf of the prior distribution evaluated at the parameter x.
        As all priors are independent the joint pdf is the product of individual pdfs.

        :param x: Particle to evaluate the pdf at
        :return: pdf value of prior distribution evaluated at x
        """
        pdf = 1.
        for index, row in self.mean.iterrows():
            pdf *= InferredInitialInfections._rvs_lognormal(row.Infected, self.stddev).pdf(x.value.at[index, "Infected"])

        return pdf

    def perturbation_pdf(self, x: InferredInitialInfections) -> float:
        """ Compute pdf of the perturbation evaluated at the parameter x,
        from the current parameter. In ABC-SMC when a particle is sampled
        from the previous population it is slightly perturbed:

        P_t* ~ K(P_t | P_{t-1}) ~ P_{t-1} + Uniform([-1,1]) * kernel_sigma

        As all perturbations are independent the joint pdf is the product
        of individual pdfs.

        :param x: Particle to evaluate the pdf at
        :return: pdf value of perturbation from previous particle evaluated at x
        """
        pdf = 1.
        for index, row in self.value.iterrows():
            pdf *= stats.uniform.pdf(x.value.at[index, "Infected"], row.Infected - self.kernel_sigma, 2 * self.kernel_sigma)

        return pdf


class Particle:
    inferred_variables_classes = {
        "infection_probability": InferredInfectionProbability,
        "initial-infections": InferredInitialInfections
    }

    def __init__(self, inferred_variables: Dict[str, InferredVariable]):
        self.inferred_variables = inferred_variables

    @staticmethod
    def generate_from_priors(fitter: ABCSMC) -> Particle:
        """ Generate a particle from the prior distribution. All parameters in the
        particle are independent and therefore we simply random each parameter from
        its prior distribution. The fitter object provides the random state used for
        random variable generation but also the prior parameters, which are
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
    def resample_and_perturbate(particles: List[Particle], weights: List[float]) -> Particle:
        """ Resampling part of ABC-SMC, selects randomly a new particle from the
        list of previously accepted. Then perturb it slightly. This causes
        the persistence and selection of fit particles in a manner similar to
        evolution algorithms. The perturbation is done until the candidate is valid,
        this is because the perturbation is unaware of the support of the distribution
        of the particle.

        :param particles: List of particles from previous population
        :param weights: List of weights of particles from previous population
        :return: Newly created particles sampled from previous population and perturbated
        """
        while True:
            particle = np.random.choice(particles, p=weights / np.sum(weights))
            particle = particle.generate_perturbated()

            if particle.validate_particle():
                return particle

    def prior_pdf(self, x: Particle) -> Optional[float]:
        """ Compute pdf of the prior distribution evaluated at the particle x.
        As all priors are independent the joint pdf is the product
        of individual pdfs.

        :param x: Particle to evaluate the pdf at
        :return: pdf value of prior distribution evaluated at x
        """
        pdf = 1.
        for key in self.inferred_variables:
            pdf *= self.inferred_variables[key].prior_pdf(x.inferred_variables[key])

        return pdf

    def perturbation_pdf(self, x: Particle) -> Optional[float]:
        """ Compute pdf of the perturbation evaluated at the particle x,
        from the current particle. In ABC-SMC when a particle is sampled
        from the previous population it is slightly perturbed:

        P_t* ~ K(P_t | P_{t-1})

        (Usually a uniform perturbation around the previous value).
        As all perturbations are independent the joint pdf is the product
        of individual pdfs.

        :param x: Particle to evaluate the pdf at
        :return: pdf value of perturbation from previous particle evaluated at x
        """
        pdf = 1.
        for key in self.inferred_variables:
            pdf *= self.inferred_variables[key].perturbation_pdf(x.inferred_variables[key])

        return pdf


class ABCSMC:

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
        start_date: pd.DataFrame,
        movement_multipliers_table: pd.DataFrame,
        stochastic_mode: pd.DataFrame,
        random_seed: pd.DataFrame
    ):
        self.historical_deaths = loaders.readHistoricalDeaths(historical_deaths)
        parameters = loaders.readABCSMCParameters(parameters)

        self.n_smc_steps = parameters["n_smc_steps"]
        self.n_particles = parameters["n_particles"]
        self.thresholds = parameters["thresholds"]

        assert self.n_smc_steps > 0
        assert self.n_particles > 0
        assert self.n_smc_steps == len(self.thresholds)

        self.compartment_transition_table = compartment_transition_table
        self.population_table = population_table
        self.commutes_table = commutes_table
        self.mixing_matrix_table = mixing_matrix_table
        self.infection_probability = infection_probability
        self.initial_infections = initial_infections
        self.infectious_states = infectious_states
        self.trials = trials
        self.start_date = start_date
        self.movement_multipliers_table = movement_multipliers_table
        self.stochastic_mode = stochastic_mode
        self.random_seed = random_seed
        self.random_state = np.random.default_rng(loaders.readRandomSeed(random_seed))

        assert trials.at[0, "Value"] == 1

        self.threshold = np.inf
        self.fit_statistics = {}

    def fit(self) -> Tuple[List[Particle], List[float]]:
        """ Performs ABC-SMC iterative procedure for finding posterior distributions
        of model parameters given priors and data. In brief, iteratively samples
        particles, keeping at each round only ones with good fitting criteria.

        References:
        https://royalsocietypublishing.org/doi/pdf/10.1098/rsif.2008.0172
        https://en.wikipedia.org/wiki/Approximate_Bayesian_computation

        :return: List of particles selected at the end of the procedure, and weights
        associated
        """
        prev_particles = []
        prev_weights = []

        for smc_step in range(self.n_smc_steps):
            logger.info(f"SMC step {smc_step + 1}/{self.n_smc_steps}")
            prev_particles, prev_weights, distances = self.sample_particles(smc_step, prev_particles, prev_weights)
            self.update_threshold(distances)

        return prev_particles, prev_weights

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
                particle = Particle.resample_and_perturbate(prev_particles, prev_weights)

            result = self.run_model(particle)
            distance = self.compute_distance(result)

            if distance <= self.threshold:
                logger.info(f"Particle accepted with distance {distance}")
                weight = ABCSMC.compute_weight(smc_step, prev_particles, prev_weights, particle)
                particles.append(particle)
                weights.append(weight)
                distances.append(distance)
                particles_accepted += 1

            particles_simulated += 1

        logger.info(f"Particles accepted {particles_accepted}/{particles_simulated}")
        self.add_iteration_statistics(smc_step, particles_accepted, particles_simulated, distances, t0)

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
            particle.inferred_variables["infection_probability"].generate_dataframe(),
            particle.inferred_variables["initial-infections"].generate_dataframe(),
            self.trials,
            self.start_date,
            self.movement_multipliers_table,
            self.stochastic_mode,
            self.random_seed
        )
        results = sm.runSimulation(network, 111)
        aggregated = sm.aggregateResults(results)
        return aggregated

    def compute_distance(self, result: pd.DataFrame) -> float:
        """ Computes distance between target and model run with current particle.
        The distance is the root mean squared distance between model and historical
        death rates:

        sqrt(mean((Target_death_rate - Model_death_rate)**2))

        :param result: Model run results
        :return: distance value between model run and target
        """
        result_by_node = (
            result
            .query("state == 'D'")
            .groupby(["date", "node"])
            .sum()
            .reset_index()
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
        """ Compute weights of particle as per the ABC-SMC algorithm

        :param smc_step: Step number of ABC-SMC algorithm
        :param particles: List of accepted particles in the previous run
        :param weights: List of weights of accepted particles in the previous run
        :param particle: Particle under consideration
        :return: Weight of the particle under consideration
        """
        if smc_step == 0:
            return 1.
        else:
            num = particle.prior_pdf(particle)
            denom = sum(weights[i] * p.perturbation_pdf(particle) for i, p in enumerate(particles))

            return num / denom

    def add_iteration_statistics(
        self,
        smc_step: int,
        particles_accepted: int,
        particles_simulated: int,
        distances: List[float],
        t0: float
    ):
        """ Log statistics of each iteration of ABC-SMC algorithm

        :param smc_step: Step number of ABC-SMC algorithm
        :param particles_accepted: Number of accepted particles
        :param particles_simulated: Number of accepted particles
        :param distances: List of distances generated by accepted particles
        :param t0: Time at start of iteration
        """
        self.fit_statistics.setdefault(smc_step, {})
        self.fit_statistics[smc_step]["particles_accepted"] = particles_accepted
        self.fit_statistics[smc_step]["particles_simulated"] = particles_simulated
        self.fit_statistics[smc_step]["distances"] = distances
        self.fit_statistics[smc_step]["threshold"] = self.threshold
        self.fit_statistics[smc_step]["time"] = f"{time.time() - t0:.0f}s"

    def summarize(self, particles: List[Particle], weights: List[float]) -> Dict:
        results = {
            "fit statistics": self.fit_statistics,
            "particles": particles,
            "weights": weights
        }

        return results


def run_inference(args) -> Dict:
    """Run inference routine

    :param args: CLI arguments
    :type args: argparse.Namespace
    :return: Result runs for inference
    """

    with data.Datastore("../config_inference.yaml") as store:
        abcsmc = ABCSMC(
            store.read_table("human/abcsmc-parameters"),
            store.read_table("human/historical-deaths"),
            store.read_table("human/compartment-transition"),
            store.read_table("human/population"),
            store.read_table("human/commutes"),
            store.read_table("human/mixing-matrix"),
            store.read_table("human/infection-probability"),
            store.read_table("human/initial-infections"),
            store.read_table("human/infectious-compartments"),
            store.read_table("human/trials"),
            store.read_table("human/start-date"),
            store.read_table("human/movement-multipliers"),
            store.read_table("human/stochastic-mode"),
            store.read_table("human/random-seed"),
        )

        t0 = time.time()
        particles, weights = abcsmc.fit()
        summary = abcsmc.summarize(particles, weights)

        logger.info("Writing output")
        store.write_table("output/simple_network_sim/inference", summary)
        logger.info("Took %.2fs to run the inference.", time.time() - t0)

    return summary


def main(argv):
    args = sm.build_args(argv)
    sm.setup_logger(args)
    logger.info("Parameters\n%s", "\n".join(f"\t{key}={value}" for key, value in args._get_kwargs()))

    run_inference(args)


if __name__ == "__main__":
    logger = logging.getLogger(f"{__package__}.{__name__}")
    main(sys.argv[1:])
