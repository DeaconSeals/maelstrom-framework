from maelstrom.population import GeneticProgrammingPopulation
from tqdm.auto import tqdm
import multiprocessing


# TODO: transition from parameters dictionary to clearer inputs with default values
class GeneticProgrammingIsland:
    """
    General-purpose island class that contains and manages multiple populations
    """

    def __init__(
        self,
        populations,
        evaluation_function,
        evaluation_kwargs={},
        population_class=GeneticProgrammingPopulation,
        initialization_kwargs={},
        eval_pool=None,
        evaluations=None,
        champions_per_generation=0,
        cores=None,
        position=None,
        **kwargs,
    ):
        """
        Initializes the island and populations based on input configuration
        parameters and evaluation function
        """
        # self.parameters = parameters
        self.populations = {}
        self.generation_count = 0
        for name, config in populations.items():
            self.populations[name] = population_class(**kwargs[config])
            self.populations[name].initialization(**initialization_kwargs)
        self.evaluation = evaluation_function

        self.evaluation_parameters = evaluation_kwargs

        self.log = {}

        if cores is None:
            cores = min(32, multiprocessing.cpu_count())
        self.cores = cores
        self.position = position

        # Fitness evaluations occur here
        with multiprocessing.Pool(self.cores) as eval_pool:
            generation_data, self.evals = self.evaluation(
                **self.populations, executor=eval_pool, **self.evaluation_parameters
            )
        for key in generation_data:
            self.log[key] = [generation_data[key]]

        self.champions_per_generation = champions_per_generation

        # identify champions for each species
        self.champions = {key: {} for key in self.populations}
        for population in self.populations:
            local_champions = self.select(
                population, self.champions_per_generation, method="best"
            )
            for individual in local_champions:
                gene_text = individual.print_tree()
                if gene_text not in self.champions[population]:
                    self.champions[population][gene_text] = individual.copy()

        self.imports = {}
        self.eval_limit = evaluations

    # Performs a single generation of evolution
    def generation(self, eval_pool=None):
        """
        Performs a single generation of evolution

        Args:
            eval_pool (multiprocessing.Pool): Pool of processes to use for evaluation

        Returns:
            self
        """
        self.generation_count += 1
        for population in self.populations:
            if population in self.imports:
                self.populations[population].generate_children(self.imports[population])
            else:
                self.populations[population].generate_children()
        self.imports.clear()

        generation_data, num_evals = self.evaluation(
            **self.populations, executor=eval_pool, **self.evaluation_parameters
        )
        self.evals += num_evals
        for key in generation_data:
            self.log[key].append(generation_data[key])

        for population in self.populations:
            self.populations[population].select_survivors()
            self.populations[population].update_hall_of_fame()

            # identify champions for each species
            local_champions = self.select(
                population, self.champions_per_generation, method="best"
            )
            for individual in local_champions:
                gene_text = individual.print_tree()
                if gene_text not in self.champions[population]:
                    self.champions[population][gene_text] = individual.copy()

        return self

    # Termination check
    def termination(self):
        """
        Checks if the island has reached termination criteria

        Returns:
            bool: True if termination criteria have been met, False otherwise
        """
        stop = False
        for key in self.populations:
            stop = stop or self.populations[key].check_termination()
            if stop:
                break
        return stop or (self.eval_limit is not None and self.evals >= self.eval_limit)

    # Selection from populations
    def select(self, population, n, method="uniform", k=5):
        """
        Selects n individuals from the specified population

        Args:
            population: Name of the population to select from
            n: Number of individuals to select
            method: Selection method to use
            k: Number of individuals to select from for tournament selection

        Returns:
            list: List of selected individuals
        """

        chosen = self.populations[population].select_unique(n, method, k)
        for index in range(len(chosen)):
            chosen[index] = chosen[index].copy()
        return chosen

    # Perfoms a single run of evolution until termination
    def run(self):
        """
        Performs a single run of evolution until termination

        Returns:
            self
        """
        with multiprocessing.Pool(self.cores) as eval_pool:
            with tqdm(
                total=self.eval_limit, unit=" evals", position=self.position
            ) as pbar:
                pbar.set_description(
                    f"COEA Generation {self.generation_count}", refresh=False
                )
                pbar.update(self.evals)
                while not self.termination():
                    evals_old = self.evals
                    # print(f"Beginning generation: {generation}\tEvaluations: {self.evals}")
                    self.generation(eval_pool)
                    pbar.set_description(
                        f"COEA Generation {self.generation_count}", refresh=False
                    )
                    pbar.update(self.evals - evals_old)
        return self  # self.log

    def build(self):
        """
        Builds the populations in the island
        """
        for population in self.populations.values():
            population.build()

    def clean(self):
        """
        Cleans the populations in the island
        """
        for population in self.populations.values():
            population.clean()
