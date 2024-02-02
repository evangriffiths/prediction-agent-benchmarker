import json

from prediction_agent_benchmarker.utils import Prediction

from examples.submodules.evo_researcher.evo_researcher.autonolas.research import (
    make_prediction,
)


def parse_prediction_str(prediction: str) -> Prediction:
    """
    Parse a prediction string of the form:

    ```json
    {
        "p_yes": 0.6,
        "p_no": 0.4,
        "confidence": 0.8,
        "info_utility": 0.9
    }
    ```

    into a Prediction object
    """
    start_index = prediction.find("{")
    end_index = prediction.rfind("}")
    prediction = prediction[start_index : end_index + 1]
    prediction_json = json.loads(prediction)
    return Prediction(
        p_yes=prediction_json["p_yes"],
        confidence=prediction_json["confidence"],
        info_utility=prediction_json["info_utility"],
    )


def _make_prediction(market_question: str, additional_information: str) -> Prediction:
    prediction: str = make_prediction(
        prompt=market_question, additional_information=additional_information
    )
    prediction: Prediction = parse_prediction_str(prediction)
    return prediction
