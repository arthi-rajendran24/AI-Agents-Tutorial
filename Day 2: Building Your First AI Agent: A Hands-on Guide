from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class MarketState:
    price: float
    volume: float
    trend: str  # 'up', 'down', or 'sideways'


class TradingAgent:
    def __init__(self):
        self.memory = []  # Store past observations
        self.performance = 0  # Track how well we're doing
        self.position = None  # Current trading position

    def sense(self, market_data: MarketState) -> Dict:
        """
        Process raw market data into useful information
        """
        observation = {
            'price': market_data.price,
            'volume': market_data.volume,
            'trend': market_data.trend,
            'price_change': self._calculate_price_change(),
            'volatility': self._calculate_volatility()
        }
        self.memory.append(observation)
        return observation

    def plan(self, observation: Dict) -> str:
        """
        Decide what action to take based on current observation
        """
        if len(self.memory) < 5:  # Need some history to make decisions
            return 'hold'

        # Simple strategy: buy on uptrend with high volume
        if (observation['trend'] == 'up' and
                observation['volume'] > self._average_volume() and
                self.position != 'long'):
            return 'buy'

        # Sell on downtrend or high volatility
        if ((observation['trend'] == 'down' or
             observation['volatility'] > self._volatility_threshold()) and
                self.position == 'long'):
            return 'sell'

        return 'hold'

    def act(self, action: str) -> None:
        """
        Execute the planned action
        """
        if action == 'buy' and self.position != 'long':
            self.position = 'long'
            print(f"📈 Buying at {self.memory[-1]['price']}")

        elif action == 'sell' and self.position == 'long':
            self.position = None
            print(f"📉 Selling at {self.memory[-1]['price']}")

        self._update_performance()

    def _calculate_price_change(self) -> float:
        if len(self.memory) < 2:
            return 0.0
        return ((self.memory[-1]['price'] - self.memory[-2]['price']) /
                self.memory[-2]['price'])

    def _calculate_volatility(self) -> float:
        if len(self.memory) < 5:
            return 0.0
        prices = [m['price'] for m in self.memory[-5:]]
        return np.std(prices)

    def _average_volume(self) -> float:
        if len(self.memory) < 5:
            return 0.0
        volumes = [m['volume'] for m in self.memory[-5:]]
        return np.mean(volumes)

    def _volatility_threshold(self) -> float:
        return 0.02  # 2% threshold, adjust based on your needs

    def _update_performance(self) -> None:
        if len(self.memory) < 2:
            return

        if self.position == 'long':
            self.performance += self._calculate_price_change()


# Test the agent
def generate_fake_market_data(n_steps: int) -> List[MarketState]:
    """Generate some fake market data for testing"""
    data = []
    price = 100.0
    for _ in range(n_steps):
        price *= (1 + np.random.normal(0, 0.02))  # 2% daily volatility
        volume = np.random.normal(1000, 200)
        trend = 'up' if price > 100 else 'down'
        data.append(MarketState(price, volume, trend))
    return data


# Create and run our agent
if __name__ == "__main__":
    agent = TradingAgent()
    market_data = generate_fake_market_data(30)  # 30 days of data

    for day, state in enumerate(market_data, 1):
        print(f"\nDay {day}")
        observation = agent.sense(state)
        action = agent.plan(observation)
        agent.act(action)
        print(f"Price: ${state.price:.2f}")
        print(f"Performance: {agent.performance:.2%}")
