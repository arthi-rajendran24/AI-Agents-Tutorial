from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
from enum import Enum
import random


class CellType(Enum):
    EMPTY = 0
    WALL = 1
    GOAL = 2
    TRAP = 3
    AGENT = 4


@dataclass
class Position:
    x: int
    y: int

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class GridWorld:
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.agent_pos = Position(0, 0)
        self.goal_pos = Position(width - 1, height - 1)
        self.traps: List[Position] = []
        self.rewards = {
            CellType.EMPTY: -0.1,  # Small penalty for each move
            CellType.GOAL: 10.0,  # Big reward for reaching goal
            CellType.TRAP: -5.0  # Big penalty for hitting trap
        }
        self.reset()

    def reset(self):
        """Reset the environment to initial state"""
        self.grid.fill(CellType.EMPTY.value)

        # Place walls (20% of grid)
        num_walls = int(0.2 * self.width * self.height)
        for _ in range(num_walls):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if Position(x, y) != self.agent_pos and Position(x, y) != self.goal_pos:
                self.grid[y, x] = CellType.WALL.value

        # Place goal
        self.grid[self.goal_pos.y, self.goal_pos.x] = CellType.GOAL.value

        # Place agent
        self.agent_pos = Position(0, 0)
        self.grid[self.agent_pos.y, self.agent_pos.x] = CellType.AGENT.value

        # Place traps
        self.traps = []
        num_traps = int(0.1 * self.width * self.height)
        for _ in range(num_traps):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pos = Position(x, y)
            if (pos != self.agent_pos and pos != self.goal_pos and
                    self.grid[y, x] == CellType.EMPTY.value):
                self.grid[y, x] = CellType.TRAP.value
                self.traps.append(pos)

        return self._get_state()

    def step(self, action: Tuple[int, int]) -> Tuple[Dict, float, bool]:
        """Take a step in the environment"""
        # Calculate new position
        new_pos = Position(
            self.agent_pos.x + action[0],
            self.agent_pos.y + action[1]
        )

        # Check if move is valid
        if not self._is_valid_position(new_pos):
            return self._get_state(), self.rewards[CellType.EMPTY], False

        # Get reward for new position
        cell_type = CellType(self.grid[new_pos.y, new_pos.x])
        reward = self.rewards.get(cell_type, 0)

        # Update grid
        self.grid[self.agent_pos.y, self.agent_pos.x] = CellType.EMPTY.value
        self.agent_pos = new_pos
        self.grid[self.agent_pos.y, self.agent_pos.x] = CellType.AGENT.value

        # Check if episode is done
        done = cell_type in [CellType.GOAL, CellType.TRAP]

        return self._get_state(), reward, done

    def _is_valid_position(self, pos: Position) -> bool:
        """Check if a position is valid"""
        if (pos.x < 0 or pos.x >= self.width or
                pos.y < 0 or pos.y >= self.height):
            return False
        return self.grid[pos.y, pos.x] != CellType.WALL.value

    def _get_state(self) -> Dict:
        """Get the current state of the environment"""
        return {
            'grid': self.grid.copy(),
            'agent_position': self.agent_pos,
            'goal_position': self.goal_pos,
            'traps': self.traps.copy()
        }

    def render(self):
        """Display the grid"""
        symbols = {
            CellType.EMPTY.value: '.',
            CellType.WALL.value: '█',
            CellType.GOAL.value: 'G',
            CellType.TRAP.value: 'X',
            CellType.AGENT.value: 'A'
        }

        print('\n' + '=' * (self.width * 2 + 1))
        for row in self.grid:
            print('|', end=' ')
            for cell in row:
                print(symbols[cell], end=' ')
            print('|')
        print('=' * (self.width * 2 + 1))


# Let's test our environment!
def test_environment():
    # Create environment
    env = GridWorld(8, 8)

    # Try some random moves
    actions = [
        (0, 1),  # Right
        (1, 0),  # Down
        (0, -1),  # Left
        (-1, 0)  # Up
    ]

    done = False
    total_reward = 0

    print("Initial state:")
    env.render()

    while not done:
        action = random.choice(actions)
        state, reward, done = env.step(action)
        total_reward += reward

        print(f"\nAction: {action}")
        print(f"Reward: {reward}")
        env.render()

        if done:
            print(f"\nGame Over! Total Reward: {total_reward}")


if __name__ == "__main__":
    test_environment()
