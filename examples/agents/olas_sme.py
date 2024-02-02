import dotenv
import os

from prediction_agent_benchmarker.utils import Prediction
from prediction_agent_benchmarker.agents import AbstractBenchmarkedAgent

# from examples.submodules.mech.tools.prediction_request_sme import prediction_request_sme
from examples.submodules.evo_researcher.evo_researcher.autonolas.research import (
    research,
)
from examples.agents.utils import _make_prediction


class OlasSMEAgent(AbstractBenchmarkedAgent):
    def __init__(self, model: str):
        super().__init__(agent_name="olas")
        self.model = model

    def research_and_predict(self, market_question: str) -> Prediction:
        ## Unable to use mech tool in same env as evo :(
        # dotenv.load_dotenv()
        # api_keys = {
        #     "openai": os.getenv("OPENAI_API_KEY"),
        #     "google_api_key": os.getenv("GOOGLE_API_KEY"),
        #     "google_engine_id": os.getenv("GOOGLE_ENGINE_ID"),
        # }
        # prediction_str = prediction_request_sme.run(
        #     tool="prediction-online-sme",
        #     api_keys=api_keys,
        #     prompt=market_question,
        #     model=self.model,
        # )
        # return parse_prediction_str(prediction_str)
        report = research(
            prompt=market_question,
            engine=self.model,
        )
        return _make_prediction(
            market_question=market_question, additional_information=report
        )
