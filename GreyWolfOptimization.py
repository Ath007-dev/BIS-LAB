import numpy as np
import random
from typing import List

# ---------------- Problem data (kept the same) ----------------
num_cars = 5
num_slots = 5
slot_distances = np.array([10, 20, 15, 30, 25])
num_wolves = 10
max_iter = 50

random.seed(0)
np.random.seed(0)


def initialize_population(num_wolves: int, num_slots: int) -> List[np.ndarray]:
    """Each wolf is a permutation (assignment) of slot indices."""
    pop = []
    for _ in range(num_wolves):
        pop.append(np.random.permutation(num_slots))
    return pop


def fitness(wolf: np.ndarray) -> float:
    """
    Total walking distance for a wolf (assignment).
    wolf[k] = slot index assigned to car k.
    Lower is better.
    """
    return float(sum(slot_distances[int(s)] for s in wolf[:num_cars]))


def repair_to_permutation(arr: List[int], n: int) -> np.ndarray:
    """
    Given a list possibly containing duplicates and missing values,
    repair it into a permutation of 0..n-1 by keeping the existing order
    for the first occurrences and appending the missing values in random order.
    """
    seen = set()
    out = []
    for x in arr:
        if x not in seen and 0 <= int(x) < n:
            out.append(int(x))
            seen.add(int(x))
    missing = [i for i in range(n) if i not in seen]
    random.shuffle(missing)
    out.extend(missing)
    return np.array(out, dtype=int)


def update_position(wolf: np.ndarray, alpha: np.ndarray, beta: np.ndarray, delta: np.ndarray, a: float) -> np.ndarray:
    """
    Build a new wolf by sampling positions from alpha/beta/delta.
    Then repair to a valid permutation and introduce small random swaps
    controlled by 'a' (exploration factor).
    """
    n = len(wolf)
    # if any of the leaders are None (first iter), fallback to random values
    if alpha is None or beta is None or delta is None:
        return np.random.permutation(n)

    candidate = []
    for i in range(n):
        r = random.random()
        if r < 1/3:
            candidate.append(int(alpha[i]))
        elif r < 2/3:
            candidate.append(int(beta[i]))
        else:
            candidate.append(int(delta[i]))

    # repair to valid permutation
    new_wolf = repair_to_permutation(candidate, n)

    # small randomization: probability to swap decreases as 'a' goes to 0.
    # normalize a in [0,2] -> an in [0,1]
    an = max(0.0, min(1.0, a / 2.0))
    swap_prob = 0.4 * an  # scale factor; keeps exploration similar to original idea
    if random.random() < swap_prob:
        i, j = random.sample(range(n), 2)
        new_wolf[i], new_wolf[j] = new_wolf[j], new_wolf[i]

    return new_wolf


def gwo_parking_allocation():
    # initialize
    population = initialize_population(num_wolves, num_slots)

    # initialize alpha/beta/delta and their scores
    alpha = beta = delta = None
    alpha_score = beta_score = delta_score = float("inf")  # minimize

    # main loop
    for iteration in range(max_iter):
        fitness_scores = []
        # evaluate
        for wolf in population:
            s = fitness(wolf)
            fitness_scores.append(s)

            # update alpha/beta/delta (best = smallest distance)
            if s < alpha_score:
                # shift down
                delta_score, delta = beta_score, beta
                beta_score, beta = alpha_score, alpha
                alpha_score, alpha = s, wolf.copy()
            elif s < beta_score:
                delta_score, delta = beta_score, beta
                beta_score, beta = s, wolf.copy()
            elif s < delta_score:
                delta_score, delta = s, wolf.copy()

        # update factor 'a' (linear decrease from 2 to 0)
        a = 2.0 - iteration * (2.0 / max_iter)

        # create new population by moving towards leaders
        new_population = []
        for wolf in population:
            new_wolf = update_position(wolf, alpha, beta, delta, a)
            new_population.append(new_wolf)

        population = new_population

        # progress print
        if (iteration + 1) % 5 == 0 or iteration == 0 or iteration == max_iter - 1:
            print(f"Iteration {iteration+1:3d}: Best total walking distance = {alpha_score:.4f}")

    # final results
    print("\nBest Assignment of Cars to Slots (car i -> slot index):")
    print(alpha)
    print("Slot distances for assignment:", slot_distances[alpha])
    print("Total walking distance:", alpha_score)


if __name__ == "__main__":
    gwo_parking_allocation()
