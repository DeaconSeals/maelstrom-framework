"""
General-purpose GP population class that contains and manages individuals
"""
import random
import math
import statistics
from collections import OrderedDict
from maelstrom.genotype import GeneticTree
# from maelstrom.individual import GeneticProgrammingIndividual


# TODO: transition from parameters dictionary to clearer inputs with default values
class GeneticProgrammingPopulation:
    """
    General-purpose GP population class that contains and manages individuals
    """

    def __init__(
        self,
        pop_size,
        num_children,
        roles,
        output_type,
        depth_limit,
        hard_limit=None,
        depth_min=1,
        evaluations=None,
        parent_selection="uniform",
        survival_selection="truncation",
        survival_strategy="plus",
        mutation=0.05,
        genotype=GeneticTree,
        **kwargs,
    ):
        """
        Initializes the population and individuals based on input configuration
        parameters and evaluation function
        """
        self.population = []
        # self.parameters = parameters
        self.pop_size = pop_size
        self.num_children = num_children
        self.genotype = genotype
        self.roles = roles
        self.output_type = output_type
        self.depth_limit = depth_limit
        self.hard_limit = hard_limit if hard_limit is not None else self.depth_limit * 2
        self.depth_min = depth_min
        self.eval_limit = evaluations
        self.evals = 0
        self.parent_selection = parent_selection
        self.survival_selection = survival_selection
        self.survival_strategy = survival_strategy
        self.mutation = mutation
        self.optional_params = kwargs
        self.hall_of_fame = OrderedDict()
        self.CIAO = []

    def ramped_half_and_half(self, leaf_prob=0.5):
        """
        Initializes the population with a ramped half-and-half method

        Args:
            leaf_prob: Probability of a leaf node when initializing a tree
        """

        full = self.pop_size // 2
        self.population = [
            self.genotype(self.roles, self.output_type)
            for _ in range(self.pop_size)
        ]

        for index in range(len(self.population)):
            if index < full:
                self.population[index].initialize(
                    self.depth_limit, self.hard_limit, full=True
                )
            else:
                self.population[index].initialize(
                    self.depth_min + (index % (self.depth_limit + 1 - self.depth_min)),
                    self.hard_limit,
                    grow=True,
                    leaf_prob=leaf_prob,
                )
    def initialization(self, *args, **kwargs):
        self.ramped_half_and_half(*args, **kwargs)

    def select_parents(self, num_parents=None):
        """
        Selects parents for the next generation based on the parent selection method

        Args:
            numParents: Number of parents to select. Defaults to None.

        Returns:
            List of parents
        """
        if num_parents == None:
            num_parents = self.num_children

        # Definitions of parent selection algorithms
        # Note: this can probably be cleaned up and segmented in a better way (maybe with the decorator approach used in genotype)
        def uniform_random(population, n):
            return random.choices(population=population, k=n)

        def k_tournament(population, n, k):
            candidates = [index for index in range(len(population))]
            winners = []
            for i in range(n):
                participants = random.sample(candidates, k)
                best = max(
                    [population[participant].fitness for participant in participants]
                )
                champion = random.choice(
                    [
                        participant
                        for participant in participants
                        if best == population[participant].fitness
                    ]
                )
                winners.append(champion)
            return [population[parent] for parent in winners]

        def fitness_proportional_selection(population, n):
            fitnesses = [individual.fitness for individual in population]
            offset = min(fitnesses)
            offset = min(0, offset)
            weights = [fitness - offset for fitness in fitnesses]
            if sum(weights) == 0:
                weights = [fitness - offset + 0.001 for fitness in fitnesses]
            return random.choices(population=population, weights=weights, k=n)

        def stochastic_universal_sampling(population, n):
            fitnesses = [individual.fitness for individual in population]
            offset = (
                min(fitnesses) * 1.1
            )  # multiply the min offset by 10% so the least fit individual has a non-zero chance of selection
            if offset == 0:
                offset = (
                    -0.01
                )  # mitigates the case where individuals with fitnesss of 0
            else:
                offset = min(0, offset)
            roulette = [fitness - offset for fitness in fitnesses]
            total = sum(roulette)
            for i in range(1, len(roulette)):
                roulette[i] = roulette[i] + roulette[i - 1]
            roulette = [value / total for value in roulette]
            parents = []
            roulette_arm = random.random()
            arms = [math.fmod(roulette_arm + (i / n), 1.0) for i in range(n)]
            arms.sort()
            pop_index = 0
            for arm in arms:
                if arm <= roulette[pop_index]:
                    parents.append(population[pop_index])
                else:
                    pop_index += 1
            random.shuffle(parents)
            return parents

        def overselection(population, n, bias=0.8, partition=10):
            if partition > len(population) or partition < 0:
                partition = math.round(0.1 * len(population))
            elites = math.round(bias * len(population))
            candidates = sorted(
                population, key=lambda individual: individual.fitness, reverse=True
            )
            parents = []
            for i in range(n):
                if i <= elites and partition > 0:
                    parents.append(random.choice(candidates[:partition]))
                else:
                    parents.append(random.choice(candidates[partition:]))
            random.shuffle(parents)
            return parents

        # Actual selection execution
        if self.parent_selection == "k_tournament":
            return k_tournament(
                self.population, num_parents, self.optional_params["k_parent"]
            )
        elif self.parent_selection == "FPS":
            return fitness_proportional_selection(self.population, num_parents)
        elif self.parent_selection == "SUS":
            return stochastic_universal_sampling(self.population, num_parents)
        elif self.parent_selection == "overselection":
            bias = self.optional_params.get("overselection_bias", 0.8)
            partition = self.optional_params.get("overselection_partition", 10)
            return overselection(self.population, num_parents, bias, partition)
        elif self.parent_selection == "uniform":
            return uniform_random(self.population, num_parents)
        else:
            raise NameError(
                f"unrecognized parent selection method: {self.parent_selection}"
            )

    # Generate children through the selection of parents, recombination or mutation of parents to form children, then the migration of children
    # into the primary population depending on survival strategy
    # TODO: generalize this so it relies on operations of the individual class instead of skipping that and working directly with the genotype
    def generate_children(self, imports=None):
        """
        Generates children for the next generation

        Args:
            imports: List of individuals to import into the population. Defaults to None.

        Raises:
            NameError: If the survival strategy is unrecognized
        """
        if imports == None:
            num_parents = self.num_children
        else:
            num_parents = max(0, self.num_children - len(imports))
        parents = self.select_parents(num_parents)
        children = [parent.copy() for parent in parents]
        for i in range(len(children)):
            if random.random() <= self.mutation:
                children[i].subtree_mutation()
            else:
                children[i].subtree_recombination(
                    children[(i + 1) % len(children)]
                )
                while children[i].depth > self.hard_limit:
                    children[i] = parents[i].copy()
                    children[i].subtree_recombination(
                        children[(i + 1) % len(children)]
                    )

        if imports != None:
            children.extend([migrant.copy() for migrant in imports])

        if self.survival_strategy == "comma":
            self.population = children
        elif self.survival_strategy == "plus":
            self.population.extend(children)
        else:
            raise NameError(f"unrecognized survival strategy: {self.survival_strategy}")

    def select_survivors(self):
        """
        Selects survivors for the next generation

        Raises:
            NameError: If the survival selection method is unrecognized
        """
        if self.survival_selection == "k_tournament":
            self.population = self.select_unique(
                n=self.pop_size,
                method="tournament",
                k=self.optional_params["k_survival"],
            )
        elif self.survival_selection == "FPS":
            self.population = self.select_unique(n=self.pop_size, method="FPS")
        elif self.survival_selection == "uniform":
            self.population = self.select_unique(n=self.pop_size, method="uniform")
        elif self.survival_selection == "truncation":
            self.population = self.select_unique(n=self.pop_size, method="truncation")
        else:
            raise NameError(
                f"unrecognized survival selection method: {self.survival_selection}"
            )

    # TODO: implement more termination methods
    def check_termination(self):
        """
        Checks if the termination condition has been met

        Returns:
            bool: True if the termination condition has been met, False otherwise
        """
        return self.eval_limit is not None and self.evals >= self.eval_limit

    def update_hall_of_fame(self):
        """
        Updates the hall of fame with the best individual in the population
        """
        best_individual = 0
        best_fitness = self.population[best_individual].fitness

        for i in range(len(self.population)):
            if self.population[i].fitness > best_fitness:
                best_individual = i
        key = self.population[best_individual].string
        self.CIAO.append(key)
        if key in self.hall_of_fame:
            self.hall_of_fame.move_to_end(key)
            return
        else:
            self.hall_of_fame[key] = self.population[best_individual].copy()

    # Selection of unique individuals for survival and migration
    # Note: this can probably be cleaned up and segmented in a better way (maybe with the decorator approach used in genotype)
    def select_unique(self, n, method="uniform", k=5):
        """
        Selects n unique individuals from the population

        Args:
            n: Number of individuals to select
            method: Selection method to use. Defaults to "uniform".
            k: Number of participants in kTournament selection. Defaults to 5.

        Raises:
            NameError: If the selection method is unrecognized

        Returns:
            List of unique individuals
        """

        # Definition of selection methods
        def uniform_random(population, n):
            return random.sample(population, n)

        def k_tournament(population, n, k):
            candidates = {index for index in range(len(population))}
            winners = []
            for i in range(n):
                participants = random.sample(list(candidates), k)
                best = max(
                    population[participant].fitness for participant in participants
                )
                champion = random.choice(
                    [
                        participant
                        for participant in participants
                        if best == population[participant].fitness
                    ]
                )
                candidates.remove(champion)
                winners.append(champion)
            return [population[survivor] for survivor in winners]

        def fitness_proportional_selection(population, n):
            offset = (
                min(individual.fitness for individual in population) * 1.1
            )  # multiply the min offset by 10% so the least fit individual isn't guaranteed to die
            if offset == 0:
                offset = (
                    -0.01
                )  # mitigates some deterministic behaviors of random.choices when all weights are 0 and avoids guaranteed death
            else:
                offset = min(0, offset)
            candidates = {index for index in range(len(population))}
            winners = []
            for i in range(n):
                champion = random.choices(
                    population=list(candidates),
                    weights=[
                        population[individual].fitness - offset
                        for individual in candidates
                    ],
                )
                candidates.remove(champion[0])
                winners.append(champion[0])
            return [population[survivor] for survivor in winners]

        def truncation(population, n):
            return sorted(
                population, key=lambda individual: individual.fitness, reverse=True
            )[:n]

        def normal_selection(population, n):
            candidates = {index for index in range(len(population))}
            winners = []
            for i in range(n):
                fitnesses = [
                    population[individual].fitness for individual in candidates
                ]
                avg = statistics.mean(fitnesses)
                dev = statistics.stdev(fitnesses)
                if fitnesses.count(0) == len(fitnesses):
                    weights = None
                else:
                    weights = [
                        dev / abs(avg - (population[individual].fitness * 1.00001))
                        for individual in candidates
                    ]
                champion = random.choices(population=list(candidates), weights=weights)
                candidates.remove(champion[0])
                winners.append(champion[0])
            return [population[survivor] for survivor in winners]

        # Execution of selection method
        if n > len(self.population):
            print("selectUnique: requested too many individuals")
            return

        if method == "tournament":
            return k_tournament(self.population, n, k)
        if method == "FPS":
            return fitness_proportional_selection(self.population, n)
        if method == "uniform" or method == "random":
            return uniform_random(self.population, n)
        if method == "normal":
            return normal_selection(self.population, n)

        if method != "truncation" and method != "best":
            print(
                f"unknown survival selection parameter '{method}' defaulting to truncation"
            )
        return truncation(self.population, n)

    def build(self):
        """Builds the population by calling the build method of each individual"""
        for individual in self.population:
            individual.build()

    def clean(self):
        """Cleans the population by calling the clean method of each individual"""
        for individual in self.population:
            individual.clean()
