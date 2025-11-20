import random
from typing import List, Tuple

# ---------- Configuration (kept the same) ----------
GRID_SIZE = 5        # grid is GRID_SIZE x GRID_SIZE
NUM_DRONES = 10      # number of drones
NUM_ITERATIONS = 20  # simulation steps
FIND_PROB = 0.20     # per-iteration probability a drone "finds something"

random.seed(0)  # remove for non-deterministic runs


class Drone:
    """Simple drone with (x,y) position and an integer fitness score."""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.fitness = 0

    def search_area(self) -> int:
        """Simulate searching: with probability FIND_PROB increment fitness."""
        if random.random() < FIND_PROB:
            self.fitness += 1
        return self.fitness

    def update_position(self, new_x: int, new_y: int) -> None:
        """Move drone to a new grid cell (coordinates clamped to grid)."""
        self.x = max(0, min(GRID_SIZE - 1, new_x))
        self.y = max(0, min(GRID_SIZE - 1, new_y))

    def pos(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Drone(x={self.x}, y={self.y}, fitness={self.fitness})"


# ---------- Utility functions ----------
def initialize_drones() -> List[Drone]:
    """Create NUM_DRONES placed uniformly at random inside the grid."""
    drones: List[Drone] = []
    for _ in range(NUM_DRONES):
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        drones.append(Drone(x, y))
    return drones


def evaluate_drones(drones: List[Drone]) -> int:
    """Run each drone's search_area() and return the total fitness (sum)."""
    total = 0
    for d in drones:
        total += d.search_area()
    return total


def get_neighbors(drone: Drone, drones: List[Drone]) -> List[Drone]:
    """
    Return drones that are neighbors of `drone`.
    Neighbor defined as Chebyshev distance <= 1 (includes diagonals),
    excluding the drone itself.
    """
    neigh: List[Drone] = []
    for other in drones:
        if other is drone:
            continue
        if abs(drone.x - other.x) <= 1 and abs(drone.y - other.y) <= 1:
            neigh.append(other)
    return neigh


def update_drone_position(drone: Drone, neighbors: List[Drone]) -> None:
    """
    If there are neighbors, move to the position of the neighbor with highest fitness.
    If multiple share the best fitness, pick one at random among them.
    If no neighbors, the drone stays put.
    """
    if not neighbors:
        return

    # find best fitness among neighbors
    best_f = max(n.fitness for n in neighbors)
    best_candidates = [n for n in neighbors if n.fitness == best_f]
    best_neighbor = random.choice(best_candidates)
    # move exactly to best neighbor's coordinates (same behavior as original)
    if (drone.x, drone.y) != (best_neighbor.x, best_neighbor.y):
        drone.update_position(best_neighbor.x, best_neighbor.y)


# ---------- Main simulation ----------
def search_and_rescue() -> int:
    drones = initialize_drones()
    best_total_fitness = 0

    # optional: show initial state
    print("Initial drones:")
    for d in drones:
        print(d)
    print("-" * 40)

    for it in range(NUM_ITERATIONS):
        # each drone searches its current cell
        total_fitness = evaluate_drones(drones)

        # track best cumulative fitness (maximize)
        if total_fitness > best_total_fitness:
            best_total_fitness = total_fitness

        # print progress
        print(f"Iteration {it + 1:2d}: Total fitness = {total_fitness}, Best so far = {best_total_fitness}")

        # update positions based on neighbors
        # note: use a copy of population so position updates don't affect neighbor calculations mid-iteration
        current_population = drones.copy()
        for drone in drones:
            neighbors = get_neighbors(drone, current_population)
            update_drone_position(drone, neighbors)

    return best_total_fitness


if __name__ == "__main__":
    best = search_and_rescue()
    print("\nBest Fitness Achieved:", best)
