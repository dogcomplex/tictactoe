# framework/strategy_registry.py
from typing import Dict, Type
import logging
from framework.strategy_interface import Strategy # Assuming Strategy interface is defined

logger = logging.getLogger(__name__)

class StrategyRegistry:
    """A registry to hold and manage different reasoning strategies."""

    def __init__(self):
        self._strategies: Dict[str, Strategy] = {}
        logger.info("StrategyRegistry initialized.")

    def register(self, strategy_instance: Strategy):
        """Registers a strategy instance."""
        strategy_id = strategy_instance.strategy_id
        if not strategy_id:
            raise ValueError("Strategy instance must have a non-empty strategy_id attribute.")
        if strategy_id in self._strategies:
            raise ValueError(f"Strategy ID '{strategy_id}' already registered.")
        self._strategies[strategy_id] = strategy_instance
        logger.info(f"Registered strategy: {strategy_id}")

    def get_strategy(self, strategy_id: str) -> Strategy:
        """Retrieves a strategy instance by its ID."""
        try:
            return self._strategies[strategy_id]
        except KeyError:
            logger.error(f"Strategy with ID '{strategy_id}' not found.")
            raise KeyError(f"Strategy with ID '{strategy_id}' not found.")

    def get_all_strategies(self) -> Dict[str, Strategy]:
        """Returns a dictionary of all registered strategy instances."""
        return self._strategies.copy()

    def setup_all(self):
        """Calls the setup() method on all registered strategies."""
        logger.info(f"Setting up {len(self._strategies)} strategies...")
        for strategy_id, strategy in self._strategies.items():
            try:
                logger.info(f"Setting up strategy: {strategy_id}")
                strategy.setup()
            except Exception as e:
                logger.error(f"Error setting up strategy {strategy_id}: {e}", exc_info=True)
        logger.info("Finished setting up strategies.")

    def teardown_all(self):
        """Calls the teardown() method on all registered strategies."""
        logger.info(f"Tearing down {len(self._strategies)} strategies...")
        for strategy_id, strategy in self._strategies.items():
            try:
                logger.info(f"Tearing down strategy: {strategy_id}")
                strategy.teardown()
            except Exception as e:
                logger.error(f"Error tearing down strategy {strategy_id}: {e}", exc_info=True)
        logger.info("Finished tearing down strategies.") 