import random
from typing import List, Sequence

# ---------- Configuration (kept same/similar defaults) ----------
POP_SIZE = 30
GENERATIONS = 200
TOURNAMENT_K = 3
MUTATION_RATE = 0.2  # probability per mutation attempt
# ---------------------------------------------------------------

def avg_waiting_time(schedule: Sequence[int], burst_times: Sequence[int]) -> float:
    """Return average waiting time for a given schedule (order of process indices)."""
    waiting = 0
    total_wait = 0
    for pid in schedule:          # each process waits the sum of bursts before it
        total_wait += waiting
        waiting += burst_times[pid]
    return total_wait / len(schedule)

def fitness(schedule: Sequence[int], burst_times: Sequence[int]) -> float:
    """Higher is better; we invert average waiting time into a fitness score."""
    return 1.0 / (1.0 + avg_waiting_time(schedule, burst_times))

def random_schedule(n: int) -> List[int]:
    s = list(range(n))
    random.shuffle(s)
    return s

def tournament_select(pop: List[List[int]], burst_times: Sequence[int], k: int = TOURNAMENT_K) -> List[int]:
    """Pick k random individuals and return the one with best fitness."""
    contenders = random.sample(pop, k)
    return max(contenders, key=lambda s: fitness(s, burst_times))

def ordered_crossover(p1: Sequence[int], p2: Sequence[int]) -> List[int]:
    """Perform ordered crossover (OX) for permutation schedules."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    # copy slice from parent1
    child[a:b] = p1[a:b]
    # fill remaining slots from parent2 in order
    fill = [x for x in p2 if x not in child]
    fi = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[fi]
            fi += 1
    return child

def swap_mutation(schedule: List[int], rate: float = MUTATION_RATE) -> List[int]:
    """Perform mutation by swapping two random genes with probability 'rate' for each attempt."""
    n = len(schedule)
    # perform n attempts (similar to your original): each attempt may swap two positions
    for _ in range(n):
        if random.random() < rate:
            i, j = random.sample(range(n), 2)
            schedule[i], schedule[j] = schedule[j], schedule[i]
    return schedule

def gene_expression_scheduler(
    burst_times: Sequence[int],
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    verbose: bool = True
) -> List[int]:
    """Run the genetic-like algorithm and return the best schedule found."""
    n = len(burst_times)
    population = [random_schedule(n) for _ in range(pop_size)]
    best = None
    best_score = -1.0

    for g in range(generations + 1):  # include final generation reporting
        # evaluate & sort population by fitness (descending)
        population.sort(key=lambda s: fitness(s, burst_times), reverse=True)

        # update best-so-far
        if best is None or fitness(population[0], burst_times) > best_score:
            best = population[0].copy()
            best_score = fitness(best, burst_times)

        # logging at selected generations (same behaviour as your original)
        if verbose and (g == 0 or g == 10 or g % 50 == 0):
            print(f"Gen {g}: Best Avg Waiting Time = {avg_waiting_time(best, burst_times):.2f}")

        # create new population
        new_pop: List[List[int]] = []
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, burst_times)
            p2 = tournament_select(population, burst_times)
            child = ordered_crossover(p1, p2)
            child = swap_mutation(child)
            new_pop.append(child)

        population = new_pop

    return best

# ---------------- Example usage ----------------
if __name__ == "__main__":
    random.seed(42)  # optional: makes results repeatable for testing

    burst_times = [5, 2, 8, 3, 6]  # P1..P5
    best_schedule = gene_expression_scheduler(burst_times)

    print("\nBest Schedule Found:", best_schedule)
    print("Avg Waiting Time:", avg_waiting_time(best_schedule, burst_times))
