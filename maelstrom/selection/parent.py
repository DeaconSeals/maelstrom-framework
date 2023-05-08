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