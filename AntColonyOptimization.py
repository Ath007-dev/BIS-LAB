import random
import math
from typing import List, Tuple

# ---------- Problem definition (same cities) ----------
cities: List[Tuple[float, float]] = [
    (0, 0),
    (1, 5),
    (5, 2),
    (6, 6),
    (8, 3),
    (7, 9),
    (2, 7),
    (3, 3),
]

N = len(cities)

# ---------- ACO parameters (kept identical to your original) ----------
NUM_ANTS = N
MAX_ITERS = 100
ALPHA = 1.0
BETA = 5.0
RHO = 0.5        # pheromone evaporation rate
Q = 100.0        # pheromone deposit factor
TAU0 = 1.0       # initial pheromone on each edge

random.seed(123)  # set seed for reproducibility (optional)

# ---------- Helper functions ----------
def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

# distance matrix
dist: List[List[float]] = [[0.0] * N for _ in range(N)]
for i in range(N):
    for j in range(N):
        dist[i][j] = euclidean(cities[i], cities[j])

# heuristic information: 1/d (avoid division by zero)
eta: List[List[float]] = [
    [0.0 if i == j else 1.0 / dist[i][j] for j in range(N)]
    for i in range(N)
]

# initialize pheromone matrix (symmetric)
tau: List[List[float]] = [[TAU0 for _ in range(N)] for _ in range(N)]

def tour_length(tour: List[int]) -> float:
    """Compute length of a tour that visits N cities and returns to start."""
    L = 0.0
    for k in range(len(tour) - 1):
        L += dist[tour[k]][tour[k + 1]]
    return L

def build_one_solution() -> List[int]:
    """Construct a single tour for an ant using probabilistic selection."""
    unvisited = list(range(N))
    current = random.choice(unvisited)
    tour = [current]
    unvisited.remove(current)

    while unvisited:
        # compute selection probabilities for every candidate next city
        probs = []
        for j in unvisited:
            t = (tau[current][j] ** ALPHA) * (eta[current][j] ** BETA)
            probs.append(t)
        total = sum(probs)
        # normalize (should not be zero because distances > 0 for i != j)
        probs = [p / total for p in probs]

        # pick next city with weights
        next_city = random.choices(unvisited, weights=probs, k=1)[0]
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    # return to start
    tour.append(tour[0])
    return tour

# ---------- Main ACO loop ----------
best_tour: List[int] = []
best_len: float = float("inf")

for iteration in range(1, MAX_ITERS + 1):
    all_tours: List[List[int]] = []
    all_lengths: List[float] = []

    # each ant builds a tour
    for _ in range(NUM_ANTS):
        t = build_one_solution()
        L = tour_length(t)
        all_tours.append(t)
        all_lengths.append(L)

        if L < best_len:
            best_len = L
            best_tour = t.copy()

    # pheromone evaporation
    for i in range(N):
        for j in range(N):
            tau[i][j] *= (1.0 - RHO)

    # pheromone deposit (symmetric)
    for tour, L in zip(all_tours, all_lengths):
        deposit = Q / L
        for k in range(len(tour) - 1):
            a, b = tour[k], tour[k + 1]
            tau[a][b] += deposit
            tau[b][a] += deposit

    # optional progress printouts (keep them sparse)
    if iteration == 1 or iteration % 10 == 0 or iteration == MAX_ITERS:
        print(f"Iteration {iteration:3d}: best length so far = {best_len:.4f}")

# ---------- Final result ----------
print("\nBest tour (indices):", best_tour)
print("Best tour length   :", round(best_len, 4))
# If you want coordinates printed:
print("Best tour (coords) :", [cities[i] for i in best_tour])
