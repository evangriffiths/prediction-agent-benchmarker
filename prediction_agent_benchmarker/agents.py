import typing as t

from prediction_agent_benchmarker.utils import Prediction


class AbstractBenchmarkedAgent:
    def __init__(self, agent_name: str, max_workers: t.Optional[int] = None):
        self.agent_name = agent_name
        self.max_workers = max_workers  # Limit the number of workers that can run this worker in parallel threads

    def research_and_predict(self, market_question: str) -> Prediction:
        raise NotImplementedError
