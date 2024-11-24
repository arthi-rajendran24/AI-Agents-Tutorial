import numpy as np


class GridWorldAgent:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.position = [0, 0]  # Start at top-left corner
        self.goal = [grid_size - 1, grid_size - 1]  # Bottom-right corner

    def sense(self):
        return {
            'position': self.position,
            'distance_to_goal': self._calculate_distance()
        }

    def think(self, state):
        # Simple strategy: move towards goal
        possible_moves = self._get_possible_moves()
        best_move = min(possible_moves,
                        key=lambda m: self._distance_after_move(m))
        return best_move

    def act(self, move):
        self.position[0] += move[0]
        self.position[1] += move[1]
        return self.position == self.goal

    def _calculate_distance(self):
        return np.sqrt((self.position[0] - self.goal[0]) ** 2 +
                       (self.position[1] - self.goal[1]) ** 2)

    def _get_possible_moves(self):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        return [m for m in moves if self._is_valid_move(m)]

    def _is_valid_move(self, move):
        new_pos = [self.position[0] + move[0], self.position[1] + move[1]]
        return (0 <= new_pos[0] < self.grid_size and
                0 <= new_pos[1] < self.grid_size)

    def _distance_after_move(self, move):
        new_pos = [self.position[0] + move[0], self.position[1] + move[1]]
        return np.sqrt((new_pos[0] - self.goal[0]) ** 2 +
                       (new_pos[1] - self.goal[1]) ** 2)


# Let's run our agent!
agent = GridWorldAgent(5)
steps = 0
goal_reached = False

while not goal_reached and steps < 100:
    state = agent.sense()
    move = agent.think(state)
    goal_reached = agent.act(move)
    steps += 1
    print(f"Step {steps}: Position {agent.position}")

print("Goal reached!" if goal_reached else "Failed to reach goal")
