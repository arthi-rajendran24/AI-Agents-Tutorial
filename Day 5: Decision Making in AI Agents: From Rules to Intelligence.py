"""
Complete implementation of the decision-making system for AI agents.
This module includes rule-based, utility-based, and pattern-aware decision makers.
"""

from enum import Enum
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import random
import numpy as np
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Action(Enum):
    MOVE_LEFT = "left"
    MOVE_RIGHT = "right"
    MOVE_UP = "up"
    MOVE_DOWN = "down"
    WAIT = "wait"

    def __str__(self):
        return self.value


@dataclass
class State:
    position: Tuple[int, int]
    energy: float
    has_goal: bool
    obstacles: List[Tuple[int, int]]
    goal_position: Optional[Tuple[int, int]] = None

    def __str__(self):
        return (f"State(pos={self.position}, energy={self.energy:.2f}, "
                f"has_goal={self.has_goal}, obstacles={len(self.obstacles)})")


@dataclass
class ActionUtility:
    action: Action
    utility: float
    confidence: float


class ActionResult:
    def __init__(self, success: bool, reward: float, new_state: State):
        self.success = success
        self.reward = reward
        self.new_state = new_state


class SimpleDecisionMaker:
    """Basic rule-based decision maker."""

    def decide(self, state: State) -> Action:
        # Safety first: avoid obstacles
        for obstacle in state.obstacles:
            if self._is_adjacent(state.position, obstacle):
                return self._avoid_obstacle(state.position, obstacle)

        # Energy management
        if state.energy < 0.2:
            return Action.WAIT

        # Goal seeking
        if state.has_goal and state.goal_position:
            return self._move_to_goal(state.position, state.goal_position)

        # Explore
        return random.choice(list(Action))

    def _is_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _avoid_obstacle(self, position: Tuple[int, int],
                        obstacle: Tuple[int, int]) -> Action:
        x_diff = position[0] - obstacle[0]
        y_diff = position[1] - obstacle[1]

        if abs(x_diff) > abs(y_diff):
            return Action.MOVE_RIGHT if x_diff < 0 else Action.MOVE_LEFT
        else:
            return Action.MOVE_DOWN if y_diff < 0 else Action.MOVE_UP

    def _move_to_goal(self, position: Tuple[int, int],
                      goal: Tuple[int, int]) -> Action:
        x_diff = goal[0] - position[0]
        y_diff = goal[1] - position[1]

        if abs(x_diff) > abs(y_diff):
            return Action.MOVE_RIGHT if x_diff > 0 else Action.MOVE_LEFT
        else:
            return Action.MOVE_UP if y_diff > 0 else Action.MOVE_DOWN


class UtilityBasedDecisionMaker:
    """Decision maker that uses utility calculations and learning."""

    def __init__(self, learning_rate: float = 0.1):
        self.memory: deque = deque(maxlen=1000)
        self.learning_rate = learning_rate

        # Initialize action utilities
        self.action_values = {
            action: {
                'baseline': 0.0,
                'success_count': 0,
                'total_count': 0
            } for action in Action
        }

    def decide(self, state: State) -> Action:
        # Get utilities for all possible actions
        utilities = self._evaluate_actions(state)

        # Apply exploration vs exploitation
        if random.random() < self._get_exploration_rate():
            logger.debug("Exploring: choosing random action")
            return random.choice(list(Action))
        else:
            logger.debug("Exploiting: choosing best known action")
            return max(utilities, key=lambda u: u.utility).action

    def _evaluate_actions(self, state: State) -> List[ActionUtility]:
        utilities = []

        for action in Action:
            # Base utility from past experience
            base_utility = self.action_values[action]['baseline']

            # Modify based on current state
            utility = base_utility

            # Safety check: avoid obstacles
            if self._would_hit_obstacle(state, action):
                utility -= 1.0

            # Energy efficiency
            if action == Action.WAIT and state.energy < 0.3:
                utility += 0.5

            # Goal seeking
            if state.has_goal and state.goal_position:
                goal_alignment = self._goal_alignment(state, action)
                utility += goal_alignment

            # Calculate confidence based on experience
            confidence = self._calculate_confidence(action)

            utilities.append(ActionUtility(action, utility, confidence))

            logger.debug(f"Action {action}: utility={utility:.2f}, "
                         f"confidence={confidence:.2f}")

        return utilities

    def _would_hit_obstacle(self, state: State, action: Action) -> bool:
        new_pos = self._get_new_position(state.position, action)
        return new_pos in state.obstacles

    def _get_new_position(self, position: Tuple[int, int],
                          action: Action) -> Tuple[int, int]:
        x, y = position
        if action == Action.MOVE_LEFT:
            return (x - 1, y)
        elif action == Action.MOVE_RIGHT:
            return (x + 1, y)
        elif action == Action.MOVE_UP:
            return (x, y + 1)
        elif action == Action.MOVE_DOWN:
            return (x, y - 1)
        return position

    def _goal_alignment(self, state: State, action: Action) -> float:
        if not state.goal_position:
            return 0.0

        current_dist = self._manhattan_distance(state.position,
                                                state.goal_position)
        new_pos = self._get_new_position(state.position, action)
        new_dist = self._manhattan_distance(new_pos, state.goal_position)

        # Return higher utility for actions that move us closer to the goal
        return (current_dist - new_dist) * 0.5

    def _manhattan_distance(self, pos1: Tuple[int, int],
                            pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _calculate_confidence(self, action: Action) -> float:
        stats = self.action_values[action]
        if stats['total_count'] == 0:
            return 0.0

        success_rate = (stats['success_count'] /
                        stats['total_count'])
        experience_weight = min(stats['total_count'] / 100, 1.0)

        return success_rate * experience_weight

    def _get_exploration_rate(self) -> float:
        total_experiences = sum(v['total_count']
                                for v in self.action_values.values())
        return max(0.1, 1.0 - (total_experiences / 1000))

    def learn(self, state: State, action: Action, result: ActionResult):
        """Learn from experience."""
        # Update action stats
        self.action_values[action]['total_count'] += 1
        if result.success:
            self.action_values[action]['success_count'] += 1

        # Update baseline utility
        current = self.action_values[action]['baseline']
        self.action_values[action]['baseline'] = (
                current + self.learning_rate * (result.reward - current)
        )

        # Store in memory
        self.memory.append((state, action, result))

        logger.debug(f"Learned from action {action}: reward={result.reward}, "
                     f"new_baseline={self.action_values[action]['baseline']:.2f}")


class PatternAwareDecisionMaker(UtilityBasedDecisionMaker):
    """Advanced decision maker that recognizes and learns from patterns."""

    def __init__(self, learning_rate: float = 0.1):
        super().__init__(learning_rate)
        self.patterns: Dict[str, List[Tuple[Action, float]]] = {}
        self.pattern_memory_size = 100

    def _recognize_pattern(self, state: State) -> str:
        """Convert state to a pattern key."""
        # Create a more detailed pattern key
        obstacle_positions = '_'.join(
            f"{x}_{y}" for x, y in sorted(state.obstacles)
        )

        pattern = (f"energy_{int(state.energy * 10)}_"
                   f"has_goal_{state.has_goal}_"
                   f"pos_{state.position[0]}_{state.position[1]}_"
                   f"obstacles_{obstacle_positions}")

        return pattern

    def _evaluate_actions(self, state: State) -> List[ActionUtility]:
        # Get basic utilities
        utilities = super()._evaluate_actions(state)

        # Modify based on recognized patterns
        pattern = self._recognize_pattern(state)
        if pattern in self.patterns:
            pattern_data = self.patterns[pattern]

            # Calculate average reward for each action in this pattern
            action_rewards = {}
            for action, reward in pattern_data:
                if action not in action_rewards:
                    action_rewards[action] = []
                action_rewards[action].append(reward)

            # Apply pattern-based modifications
            for action_utility in utilities:
                if action_utility.action in action_rewards:
                    rewards = action_rewards[action_utility.action]
                    avg_reward = sum(rewards) / len(rewards)
                    recency_weight = len(rewards) / self.pattern_memory_size

                    # Boost utility based on pattern success
                    action_utility.utility += avg_reward * recency_weight
                    action_utility.confidence += 0.1 * recency_weight

                    logger.debug(f"Pattern modification for {action_utility.action}: "
                                 f"avg_reward={avg_reward:.2f}, "
                                 f"weight={recency_weight:.2f}")

        return utilities

    def learn(self, state: State, action: Action, result: ActionResult):
        super().learn(state, action, result)

        # Update pattern memory
        pattern = self._recognize_pattern(state)
        if pattern not in self.patterns:
            self.patterns[pattern] = []

        self.patterns[pattern].append((action, result.reward))

        # Trim pattern memory
        if len(self.patterns[pattern]) > self.pattern_memory_size:
            self.patterns[pattern] = self.patterns[pattern][-self.pattern_memory_size:]

        logger.debug(f"Updated pattern {pattern}: {len(self.patterns[pattern])} "
                     f"experiences")


class Environment:
    """Simple environment for testing decision makers."""

    def __init__(self, width: int = 10, height: int = 10,
                 num_obstacles: int = 5):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self) -> State:
        """Reset the environment and return initial state."""
        self.agent_pos = (0, 0)
        self.agent_energy = 1.0
        self.goal_pos = (self.width - 1, self.height - 1)

        # Generate random obstacles
        self.obstacles = []
        while len(self.obstacles) < self.num_obstacles:
            pos = (random.randint(0, self.width - 1),
                   random.randint(0, self.height - 1))
            if pos != self.agent_pos and pos != self.goal_pos:
                self.obstacles.append(pos)

        return self._get_state()

    def step(self, action: Action) -> Tuple[State, float, bool]:
        """Execute action and return (new_state, reward, done)."""
        # Update position
        new_pos = self._get_new_position(self.agent_pos, action)

        # Check if move is valid
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos

        # Update energy
        energy_cost = 0.1 if action != Action.WAIT else 0.0
        self.agent_energy = max(0.0, self.agent_energy - energy_cost)
        if action == Action.WAIT:
            self.agent_energy = min(1.0, self.agent_energy + 0.2)

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check if done
        done = (self.agent_pos == self.goal_pos or
                self.agent_energy <= 0.0)

        return self._get_state(), reward, done

    def _get_state(self) -> State:
        return State(
            position=self.agent_pos,
            energy=self.agent_energy,
            has_goal=True,
            obstacles=self.obstacles,
            goal_position=self.goal_pos
        )

    def _get_new_position(self, position: Tuple[int, int],
                          action: Action) -> Tuple[int, int]:
        x, y = position
        if action == Action.MOVE_LEFT:
            return (max(0, x - 1), y)
        elif action == Action.MOVE_RIGHT:
            return (min(self.width - 1, x + 1), y)
        elif action == Action.MOVE_UP:
            return (x, min(self.height - 1, y + 1))
        elif action == Action.MOVE_DOWN:
            return (x, max(0, y - 1))
        return position

    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        return position not in self.obstacles

    def _calculate_reward(self, action: Action) -> float:
        if self.agent_pos == self.goal_pos:
            return 10.0
        if self.agent_energy <= 0.0:
            return -5.0
        if action == Action.WAIT:
            return -0.1
        return -0.1  # Small penalty for moving


def test_decision_makers():
    """Test and compare different decision makers."""
    env = Environment(width=5, height=5, num_obstacles=3)

    decision_makers = {
        'Simple': SimpleDecisionMaker(),
        'Utility': UtilityBasedDecisionMaker(),
        'Pattern': PatternAwareDecisionMaker()
    }

    results = {name: [] for name in decision_makers}
    num_episodes = 100

    for name, dm in decision_makers.items():
        print(f"\nTesting {name} Decision Maker:")

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done and steps < 100:
                # Get action from decision maker
                action = dm.decide(state)

                # Execute action
                new_state, reward, done = env.step(action)
                total_reward += reward

                # Learn from the experience (if supported)
                if hasattr(dm, 'learn'):
                    result = ActionResult(
                        success=(reward > 0),
                        reward=reward,
                        new_state=new_state
                    )
                    dm.learn(state, action, result)

                state = new_state
                steps += 1

            results[name].append(total_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(results[name][-10:])
                print(f"Episode {episode + 1}: Average Reward = {avg_reward:.2f}")

    return results


def plot_results(results: Dict[str, List[float]]):
    """Plot the learning curves for different decision makers."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        for name, rewards in results.items():
            # Calculate moving average
            moving_avg = [np.mean(rewards[max(0, i - 10):i + 1])
                          for i in range(len(rewards))]
            plt.plot(moving_avg, label=name)

        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Decision Maker Performance Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    except ImportError:
        print("Matplotlib not available for plotting")
        # Print summary statistics instead
        for name, rewards in results.items():
            print(f"\n{name} Decision Maker:")
            print(f"Final Average Reward: {np.mean(rewards[-10:]):.2f}")
            print(f"Best Reward: {max(rewards):.2f}")
            print(f"Worst Reward: {min(rewards):.2f}")


class StrategyAgent:
    """An agent that can switch between different strategies."""

    def __init__(self):
        self.decision_maker = PatternAwareDecisionMaker()
        self.strategies = {
            'aggressive': lambda u: u * 1.5 if u > 0 else u,
            'cautious': lambda u: u * 0.5 if u > 0 else u * 2,
            'balanced': lambda u: u
        }
        self.current_strategy = 'balanced'
        self.strategy_performance = {
            strategy: deque(maxlen=10) for strategy in self.strategies
        }

    def set_strategy(self, strategy: str):
        """Change the current strategy."""
        if strategy in self.strategies:
            self.current_strategy = strategy
            logger.info(f"Switched to {strategy} strategy")

    def decide(self, state: State) -> Action:
        """Make a decision using the current strategy."""
        utilities = self.decision_maker._evaluate_actions(state)

        # Apply strategy modifier
        for u in utilities:
            u.utility = self.strategies[self.current_strategy](u.utility)

        return max(utilities, key=lambda u: u.utility).action

    def learn(self, state: State, action: Action, result: ActionResult):
        """Learn from experience and update strategy performance."""
        self.decision_maker.learn(state, action, result)
        self.strategy_performance[self.current_strategy].append(result.reward)

        # Potentially switch strategy based on performance
        self._evaluate_strategy_switch()

    def _evaluate_strategy_switch(self):
        """Consider switching strategies based on recent performance."""
        if len(self.strategy_performance[self.current_strategy]) < 5:
            return

        current_avg = np.mean(self.strategy_performance[self.current_strategy])
        best_strategy = self.current_strategy
        best_avg = current_avg

        for strategy, performance in self.strategy_performance.items():
            if len(performance) >= 5:
                avg = np.mean(performance)
                if avg > best_avg:
                    best_avg = avg
                    best_strategy = strategy

        if best_strategy != self.current_strategy:
            self.set_strategy(best_strategy)


class MetaLearningAgent:
    """An agent that can switch between different decision makers."""

    def __init__(self):
        self.decision_makers = {
            'utility': UtilityBasedDecisionMaker(),
            'pattern': PatternAwareDecisionMaker()
        }
        self.performance_history = {
            name: deque(maxlen=100) for name in self.decision_makers
        }
        self.current_dm = 'pattern'

    def decide(self, state: State) -> Action:
        """Choose the best performing decision maker and use it."""
        return self.decision_makers[self.current_dm].decide(state)

    def learn(self, state: State, action: Action, result: ActionResult):
        """Learn from experience and update decision maker selection."""
        # Let all decision makers learn
        for dm in self.decision_makers.values():
            dm.learn(state, action, result)

        # Update performance history
        self.performance_history[self.current_dm].append(result.reward)

        # Potentially switch decision makers
        self._evaluate_dm_switch()

    def _evaluate_dm_switch(self):
        """Consider switching decision makers based on recent performance."""
        if all(len(hist) >= 10 for hist in self.performance_history.values()):
            performances = {
                name: np.mean(list(hist)[-10:])
                for name, hist in self.performance_history.items()
            }
            best_dm = max(performances.items(), key=lambda x: x[1])[0]

            if best_dm != self.current_dm:
                logger.info(f"Switching decision maker from {self.current_dm} "
                            f"to {best_dm}")
                self.current_dm = best_dm


def main():
    """Run a comprehensive test of all decision makers."""
    # Test basic decision makers
    print("Testing basic decision makers...")
    results = test_decision_makers()
    plot_results(results)

    # Test strategy agent
    print("\nTesting Strategy Agent...")
    env = Environment(width=5, height=5, num_obstacles=3)
    agent = StrategyAgent()

    total_rewards = []

    for episode in range(50):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.decide(state)
            new_state, reward, done = env.step(action)
            result = ActionResult(success=(reward > 0), reward=reward,
                                  new_state=new_state)
            agent.learn(state, action, result)
            state = new_state
            total_reward += reward

        total_rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: "
                  f"Strategy: {agent.current_strategy}, "
                  f"Reward: {total_reward:.2f}")

    # Test meta-learning agent
    print("\nTesting Meta-Learning Agent...")
    agent = MetaLearningAgent()

    meta_rewards = []

    for episode in range(50):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.decide(state)
            new_state, reward, done = env.step(action)
            result = ActionResult(success=(reward > 0), reward=reward,
                                  new_state=new_state)
            agent.learn(state, action, result)
            state = new_state
            total_reward += reward

        meta_rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: "
                  f"Decision Maker: {agent.current_dm}, "
                  f"Reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
    # Create an environment and agent
    env = Environment(width=5, height=5, num_obstacles=3)
    agent = PatternAwareDecisionMaker()

    # Run an episode
    state = env.reset()
    done = False
    while not done:
        action = agent.decide(state)
        new_state, reward, done = env.step(action)
        result = ActionResult(success=(reward > 0), reward=reward,
                              new_state=new_state)
        agent.learn(state, action, result)
        state = new_state
