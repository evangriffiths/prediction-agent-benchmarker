import dotenv
import os

from prediction_agent_benchmarker.utils import Prediction
from prediction_agent_benchmarker.agents import AbstractBenchmarkedAgent

from examples.submodules.evo_researcher.evo_researcher.functions.research import (
    research,
)
from examples.agents.utils import _make_prediction


class EvoAgent(AbstractBenchmarkedAgent):
    def __init__(self, model: str):
        super().__init__(agent_name="evo", max_workers=4)
        self.model = model

    def research_and_predict(self, market_question: str) -> Prediction:
        dotenv.load_dotenv()
        open_ai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        report, _ = research(
            goal=market_question,
            openai_key=open_ai_key,
            tavily_key=tavily_key,
            model=self.model,
        )
        return _make_prediction(
            market_question=market_question, additional_information=report
        )
