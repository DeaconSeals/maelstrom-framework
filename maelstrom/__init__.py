"""
Maelstrom Framework
High-level guiding principles:
- Classes are defined hierarchically and rely on wrapper/interface
functions to interact down the hierarchy
- Fitness evaluation is implemented as an external function and the
function itself it passed to the island
- Fitness evaluations are expected to accept named
GeneticProgrammingPopulation objects as input and assign fitness to the individuals
- This file is agnostic of the nodes used in evolution
- This framework is designed with coevolution in mind, but one could easily
use a single-population island with an appropriate fitness function

"""

import multiprocessing
from tqdm.auto import tqdm

# import concurrent.futures
from maelstrom.island import GeneticProgrammingIsland


# General-purpose Maelstrom class that contains and manages multiple islands
class Maelstrom:
    """
    Class that handles coevolutionary evolution of multiple populations
    """

    def __init__(
        self,
        islands: dict,
        island_class=GeneticProgrammingIsland,
        # TODO: do we really want this to default to None instead of a #?
        evaluations=None,
        # TODO: do we want to default this to None instead of throwing err?
        migration_edges=None,
        cores=None,
        position=None,
        **kwargs,
    ):
        """
        Initializes a Maelstrom object

        Args:
            islands: dictionary of island names and island parameters
            evaluations: total number of evaluations to perform
            migration_edges: list of migration edges
            cores: number of cores to use
            position: position of progress bar
            **kwargs: keyword arguments to pass to island initialization
        """
        self.islands = {}
        self.island_class = island_class
        self.migration_edges = migration_edges
        self.evals = 0
        self.eval_limit = evaluations
        self.log = {}
        if cores is None:
            cores = min(32, multiprocessing.cpu_count())
        self.cores = cores
        self.position = position
        # self.evalPool = multiprocessing.Pool(cores)

        # Initialize islands
        for key in islands:
            self.islands[key] = self.island_class(
                cores=self.cores, **kwargs[islands[key]], **kwargs
            )
        self.evals = sum(island.evals for island in self.islands.values())

        self.champions = {}

    # def __del__(self):
    # 	self.evalPool.close()

    def run(self):
        """
        Performs a single run of evolution until termination
        """
        generation = 1

        self.evals = sum(island.evals for island in self.islands.values())
        with multiprocessing.Pool(self.cores) as eval_pool:
            with tqdm(
                total=self.eval_limit, unit=" evals", position=self.position
            ) as pbar:
                pbar.set_description(
                    f"Maelstrom Generation {generation}", refresh=False
                )
                pbar.update(self.evals)
                while self.evals < self.eval_limit:
                    evals_old = self.evals
                    # print(f"Beginning generation: {generation}\tEvaluations: {self.evals}")

                    # migration
                    for edge in self.migration_edges:
                        # check migration timing
                        if generation % edge["period"] == 0:
                            destinationIsland, destinationPopulation = edge[
                                "destination"
                            ]
                            sourceIsland, sourcePopulation = edge["source"]
                            # collect imports
                            migrants = self.islands[sourceIsland].select(
                                population=sourcePopulation,
                                n=edge["size"],
                                method=edge["method"],
                            )
                            # export to destimation population
                            if (
                                destinationPopulation
                                in self.islands[destinationIsland].imports
                            ):
                                self.islands[destinationIsland].imports[
                                    destinationPopulation
                                ].extend(migrants)
                            else:
                                self.islands[destinationIsland].imports[
                                    destinationPopulation
                                ] = migrants

                    # Evolve one full generation with each island
                    with multiprocessing.pool.ThreadPool() as executor:
                        executor.starmap(
                            self.island_class.generation,
                            [(island, eval_pool) for island in self.islands.values()],
                        )
                    self.evals = sum(island.evals for island in self.islands.values())
                    generation += 1
                    pbar.set_description(
                        f"Maelstrom Generation {generation}", refresh=False
                    )
                    pbar.update(self.evals - evals_old)

                    island_termination = False
                    for _, island in self.islands.items():
                        island_termination = island_termination or island.termination()
                    if island_termination:
                        break

        # identify champions for each species on each island
        for _, island in self.islands.items():
            for species, champions in island.champions.items():
                if species not in self.champions:
                    self.champions[species] = {}
                self.champions[species].update(champions)

        for key, val in self.islands.items():
            self.log[key] = val.log
        return self

    def build(self):
        """
        Builds islands
        """
        for island in self.islands.values():
            island.build()

    def clean(self):
        """
        Cleans islands
        """
        for island in self.islands.values():
            island.clean()
