import random
import string
from typing import List

# Problem settings (kept the same as your original)
TARGET_PHRASE = "HELLO WORLD"
POPULATION_SIZE = 100
MUT_RATE = 0.01
MAX_GENERATIONS = 1000
ALPHABET = string.ascii_uppercase + " "  # allowed characters

def score(individual: str) -> int:
    """Fitness: number of characters matching the target at same positions."""
    return sum(a == b for a, b in zip(individual, TARGET_PHRASE))

def create_member() -> str:
    """Make a random string of the same length as the target."""
    return ''.join(random.choice(ALPHABET) for _ in range(len(TARGET_PHRASE)))

def apply_mutation(member: str) -> str:
    """Mutate each character with probability MUT_RATE."""
    out = []
    for ch in member:
        if random.random() < MUT_RATE:
            out.append(random.choice(ALPHABET))
        else:
            out.append(ch)
    return ''.join(out)

def mate(parent1: str, parent2: str) -> str:
    """Single-point crossover. Crossover point chosen from 0..len-1 (same behaviour)."""
    point = random.randint(0, len(TARGET_PHRASE) - 1)
    return parent1[:point] + parent2[point:]

def tournament_select(pop: List[str], k: int = 5) -> str:
    """Select the best out of k random individuals (tournament of size 5 by default)."""
    contenders = random.sample(pop, k)
    return max(contenders, key=score)

def run_genetic():
    # initial population
    population = [create_member() for _ in range(POPULATION_SIZE)]

    for generation in range(MAX_GENERATIONS):
        # sort by fitness (highest first)
        population.sort(key=score, reverse=True)
        best = population[0]
        print(f"Gen {generation}: {best} (Fitness = {score(best)})")

        if best == TARGET_PHRASE:
            print("Target reached!")
            break

        # elitism: carry top 2 to next generation unchanged
        next_pop = population[:2]

        # fill remaining population
        while len(next_pop) < POPULATION_SIZE:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            child = mate(p1, p2)
            child = apply_mutation(child)
            next_pop.append(child)

        population = next_pop

    # final output
    population.sort(key=score, reverse=True)
    print("\nBest match after run:", population[0], "(Fitness =", score(population[0]), ")")

if __name__ == "__main__":
    run_genetic()
