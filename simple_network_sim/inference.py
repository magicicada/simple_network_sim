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
    perturbation_scale: float = 0.1

    def __init__(self, value: float, mean: float, location: float, random_state: np.random.Generator):
        self.value = value
        self.mean = mean
        self.location = location
        self.random_state = random_state

    @staticmethod
    def generate_from_prior(fitter: ABCSMC) -> InferredInfectionProbability:
        location = 4.
        mean = fitter.infection_probability.at[0, "Value"]
        value = stats.beta.rvs(location, location * (1 - mean) / mean, random_state=fitter.random_state)
        return InferredInfectionProbability(value, mean, location, fitter.random_state)

    def generate_perturbated(self) -> InferredInfectionProbability:
        sigma = InferredInfectionProbability.perturbation_scale
        perturbated_value = self.value + stats.uniform.rvs(-1., 2., random_state=self.random_state) * sigma
        return InferredInfectionProbability(perturbated_value, self.mean, self.location, self.random_state)

    def validate(self) -> bool:
        return 0. < self.value < 1.

    def generate_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([0., self.value], columns=[0], index=["Time", "Value"]).T

    def prior_pdf(self, x: InferredInfectionProbability) -> float:
        return stats.beta.pdf(x.value, self.location, self.location * (1 - self.mean) / self.mean)

    def perturbation_pdf(self, x: InferredInfectionProbability) -> float:
        sigma = InferredInfectionProbability.perturbation_scale
        return stats.uniform.pdf(x.value, self.value - sigma, 2 * sigma)


class InferredInitialInfections(InferredVariable):
    perturbation_scale: float = 5.

    def __init__(self, value: pd.DataFrame, mean: pd.DataFrame, stddev: float, random_state: np.random.Generator):
        self.value = value
        self.mean = mean
        self.stddev = stddev
        self.random_state = random_state

    @staticmethod
    def _rvs_lognormal_from_mean_std(mean: float, stddev: float):
        mean = max(mean, 1e-5)
        var = max((mean * stddev)**2, 100.)  # Minimal stddev of 10 people for initial infections
        sigma = np.sqrt(np.log(1 + (var / mean**2)))
        mu = np.log(mean / np.sqrt(1 + (var / mean**2)))
        return stats.lognorm(s=sigma, loc=0., scale=np.exp(mu))

    @staticmethod
    def generate_from_prior(fitter: ABCSMC) -> InferredInitialInfections:
        stddev = 0.2

        value = fitter.initial_infections.copy()
        value.Infected = value.Infected.apply(lambda x: InferredInitialInfections._rvs_lognormal_from_mean_std(x, stddev)
                                              .rvs(random_state=fitter.random_state))
        return InferredInitialInfections(value, fitter.initial_infections, stddev, fitter.random_state)

    def generate_perturbated(self) -> InferredInitialInfections:
        sigma = InferredInfectionProbability.perturbation_scale

        perturbated_value = self.value.copy()
        perturbated_value.Infected = perturbated_value.Infected.apply(lambda x: x + sigma * stats.uniform
                                                                      .rvs(-1., 2., random_state=self.random_state))

        return InferredInitialInfections(perturbated_value, self.mean, self.stddev, self.random_state)

    def validate(self) -> bool:
        return np.all(self.value.Infected >= 0.)

    def generate_dataframe(self) -> pd.DataFrame:
        return self.value

    def prior_pdf(self, x: InferredInitialInfections) -> float:
        pdf = 1.
        for index, row in self.mean.iterrows():
            pdf *= InferredInitialInfections._rvs_lognormal_from_mean_std(row.Infected, self.stddev)\
                   .pdf(x.value.at[index, "Infected"])

        return pdf

    def perturbation_pdf(self, x: InferredInfectionProbability) -> float:
        sigma = InferredInfectionProbability.perturbation_scale

        pdf = 1.
        for index, row in self.value.iterrows():
            pdf *= stats.uniform.pdf(x.value.at[index, "Infected"], row.Infected - sigma, 2 * sigma)

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
        return Particle({variable_name: variable_class.generate_from_prior(fitter)
                         for variable_name, variable_class in Particle.inferred_variables_classes.items()})

    def generate_perturbated(self) -> Particle:
        perturbed_variables = {}
        for name, inferred_variable in self.inferred_variables.items():
            perturbed_variables[name] = inferred_variable.generate_perturbated()

        return Particle(perturbed_variables)

    def validate_particle(self) -> bool:
        return all(variable.validate() for variable in self.inferred_variables.values())

    @staticmethod
    def resample_and_perturbate(particles: List[Particle], weights: List[float]) -> Particle:
        while True:
            particle = np.random.choice(particles, p=weights / np.sum(weights))
            particle = particle.generate_perturbated()

            if particle.validate_particle():
                return particle

    def prior_pdf(self, at: Particle) -> Optional[float]:
        pdf = 1.
        for key in self.inferred_variables:
            pdf *= self.inferred_variables[key].prior_pdf(at.inferred_variables[key])

        return pdf

    def perturbation_pdf(self, at: Particle) -> Optional[float]:
        pdf = 1.
        for key in self.inferred_variables:
            pdf *= self.inferred_variables[key].perturbation_pdf(at.inferred_variables[key])

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
        prev_particles = []
        prev_weights = []

        for smc_step in range(self.n_smc_steps):
            logger.info(f"SMC step {smc_step + 1}/{self.n_smc_steps}")
            prev_particles, prev_weights, distances = self.sample_particles(smc_step, prev_particles, prev_weights)
            self.update_threshold(distances)

        return prev_particles, prev_weights

    def update_threshold(self, distances: List[float]):
        self.threshold = np.percentile(distances, 50)

    def sample_particles(
        self,
        smc_step: int,
        prev_particles: List[Particle],
        prev_weights: List[float]
    ) -> Tuple[List[Particle], List[float], List[float]]:

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
                weight = self.compute_weight(smc_step, prev_particles, prev_weights, particle)
                particles.append(particle)
                weights.append(weight)
                distances.append(distance)
                particles_accepted += 1

            particles_simulated += 1

        logger.info(f"Particles accepted {particles_accepted}/{particles_simulated}")
        self.add_iteration_statistics(smc_step, particles_accepted, particles_simulated, distances, t0)

        return particles, weights, distances

    def run_model(self, particle: Particle) -> pd.DataFrame:
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

    def compute_weight(
        self,
        smc_step: int,
        particles: List[Particle],
        weights: List[float],
        particle: Particle
    ) -> float:
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


def run_inference(args, store):
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

    particles, weights = abcsmc.fit()
    summary = abcsmc.summarize(particles, weights)

    return summary


def main(argv):
    t0 = time.time()

    args = sm.build_args(argv)
    sm.setup_logger(args)
    logger.info("Parameters\n%s", "\n".join(f"\t{key}={value}" for key, value in args._get_kwargs()))

    with data.Datastore("../config_inference.yaml") as store:
        results = run_inference(args, store)

        logger.info("Writing output")
        store.write_table("output/simple_network_sim/inference", results)
        logger.info("Took %.2fs to run the inference.", time.time() - t0)


if __name__ == "__main__":
    logger = logging.getLogger(f"{__package__}.{__name__}")
    main(sys.argv[1:])
