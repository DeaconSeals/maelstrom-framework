def uniform_random(population, n):
    assert n <= len(population)
    return random.sample(population, n)

def k_tournament(population, n, k):
    assert n <= len(population)
    assert len(population)-n <= k
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
    assert n <= len(population)
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

def truncation(population, n, copy=False):
    assert n <= len(population)
    if copy:
        return sorted(
            population[:], key=lambda individual: individual.fitness, reverse=True
        )[:n]    
    else:
        return sorted(
            population, key=lambda individual: individual.fitness, reverse=True
        )[:n]

def normal_selection(population, n):
    assert n <= len(population)
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