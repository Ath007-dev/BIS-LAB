import random
import numpy as np
from typing import List, Tuple

# ---------------- Problem data (unchanged) ----------------
VALUES = [60, 100, 120, 80, 30]
WEIGHTS = [10, 20, 30, 15, 5]
MAX_WEIGHT = 50
NUM_ITEMS = len(VALUES)

# ---------------- Algorithm parameters (unchanged) ----------------
NUM_NESTS = 25
MAX_ITERS = 100
DISCOVERY_RATE = 0.25   # fraction of worst nests to abandon
FLIP_PROB = 0.30        # per-bit flip probability used in the "Lévy" step

random.seed(1)  # optional: make runs repeatable


def evaluate(nest: List[int]) -> int:
    """
    Fitness: total value of selected items, or 0 if weight limit is exceeded.
    The algorithm maximizes this fitness.
    """
    total_w = sum(w for bit, w in zip(nest, WEIGHTS) if bit)
    if total_w > MAX_WEIGHT:
        return 0
    return sum(v for bit, v in zip(nest, VALUES) if bit)


def make_random_nest() -> List[int]:
    """Return a random binary solution (0/1 list of length NUM_ITEMS)."""
    return [random.randint(0, 1) for _ in range(NUM_ITEMS)]


def levy_flight_like_flip(nest: List[int]) -> List[int]:
    """
    Simple binary 'Lévy' step: flip each bit with FLIP_PROB.
    (Kept the ~30% flip chance from your original.)
    """
    child = nest.copy()
    for i in range(NUM_ITEMS):
        if random.random() < FLIP_PROB:
            child[i] = 1 - child[i]
    return child


def abandon_worst(pop: List[List[int]], fitnesses: List[int]) -> None:
    """
    Replace the worst DISCOVERY_RATE fraction of nests with random solutions.
    This modifies `pop` in place.
    """
    k = int(DISCOVERY_RATE * NUM_NESTS)
    if k <= 0:
        return
    # indices of k smallest fitness values
    worst_idx = np.argsort(fitnesses)[:k]
    for idx in worst_idx:
        pop[idx] = make_random_nest()


def cuckoo_knapsack() -> Tuple[List[int], int]:
    """
    Run the metaheuristic and return (best_solution, best_fitness).
    Behavior preserved from your original script.
    """
    # initialize nests
    nests = [make_random_nest() for _ in range(NUM_NESTS)]
    fitnesses = [evaluate(n) for n in nests]

    best_nest = nests[int(np.argmax(fitnesses))].copy()
    best_score = max(fitnesses)

    for it in range(1, MAX_ITERS + 1):
        # For each nest, try a Lévy-like candidate and accept if better
        for i in range(NUM_NESTS):
            candidate = levy_flight_like_flip(nests[i])
            cand_f = evaluate(candidate)

            # greedy replacement
            if cand_f > fitnesses[i]:
                nests[i] = candidate
                fitnesses[i] = cand_f

                # update global best if needed
                if cand_f > best_score:
                    best_score = cand_f
                    best_nest = candidate.copy()

        # abandon a fraction of worst nests and replace with fresh random nests
        abandon_worst(nests, fitnesses)

        # re-evaluate after abandonment
        fitnesses = [evaluate(n) for n in nests]

        # (Optional) keep track of best after re-evaluation
        current_best_idx = int(np.argmax(fitnesses))
        if fitnesses[current_best_idx] > best_score:
            best_score = fitnesses[current_best_idx]
            best_nest = nests[current_best_idx].copy()

        # sparse progress reporting (mirrors typical debug prints)
        if it == 1 or it % 10 == 0 or it == MAX_ITERS:
            print(f"Iter {it:3d} — best fitness so far: {best_score}")

    return best_nest, best_score


if __name__ == "__main__":
    best_solution, best_val = cuckoo_knapsack()
    total_w = sum(WEIGHTS[i] for i in range(NUM_ITEMS) if best_solution[i] == 1)

    print("\nBest Solution (0/1 per item):", best_solution)
    print("Best Total Value:", best_val)
    print("Total Weight:", total_w)
