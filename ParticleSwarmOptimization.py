import numpy as np

# --- Objective (same as your original) ---
def objective(x: np.ndarray) -> np.ndarray:
    # returns elementwise values for an array of positions
    return -x**2 + 20*x + 5

# --- Initial state (kept identical) ---
pos = np.array([0.6, 2.3, 2.8, 8.3, 10.0, 9.6, 6.0, 2.6, 1.1], dtype=float)
vel = np.zeros_like(pos)

# personal bests (positions and their scores)
pbest_pos = pos.copy()
pbest_score = objective(pbest_pos)

# global best (position with highest personal-best score)
gbest_pos = pbest_pos[np.argmax(pbest_score)]

# PSO hyper-parameters (same values)
c1 = 1.0
c2 = 1.0
w = 1.0

# random pairs supplied by your example (used per iteration)
r_pairs = [
    (0.213, 0.876),
    (0.113, 0.706),
    (0.178, 0.507),
]

# --- Print initial diagnostics ---
print("Initial positions:\n", pos)
print("Initial function values:\n", objective(pos))
print("Initial global best position:", gbest_pos, "with value:", objective(gbest_pos))
print("-" * 50)

# --- Run exactly 3 iterations using the provided r values ---
for t in range(3):
    r1, r2 = r_pairs[t]

    # velocity update (vectorized loop for clarity and exact per-particle formula)
    for i in range(pos.size):
        vel[i] = (
            w * vel[i]
            + c1 * r1 * (pbest_pos[i] - pos[i])
            + c2 * r2 * (gbest_pos - pos[i])
        )

    # position update
    pos = pos + vel

    # evaluate objective at new positions
    vals = objective(pos)

    # update personal bests
    for i in range(pos.size):
        if vals[i] > pbest_score[i]:
            pbest_score[i] = vals[i]
            pbest_pos[i] = pos[i]

    # update global best
    gbest_pos = pbest_pos[np.argmax(pbest_score)]

    # display iteration results
    print(f"Iteration {t + 1}")
    print("Positions:", np.round(pos, 4))
    print("Velocities:", np.round(vel, 4))
    print("Function values:", np.round(vals, 4))
    print("Global best position:", gbest_pos, "with value:", objective(gbest_pos))
    print("-" * 50)
