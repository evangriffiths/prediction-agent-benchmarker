from examples.agents.evo import EvoAgent
from examples.agents.olas_sme import OlasSMEAgent

from prediction_agent_benchmarker.benchmark import Benchmarker
from prediction_agent_benchmarker.utils import MarketSource, get_markets


if __name__ == "__main__":
    benchmarker = Benchmarker(
        markets=get_markets(number=1, source=MarketSource.MANIFOLD),
        agents=[
            OlasSMEAgent(model="gpt-3.5-turbo"),
            EvoAgent(model="gpt-4-1106-preview"),
        ],
    )
    benchmarker.run_agents()
    md = benchmarker.generate_markdown_report()

    output = "./benchmark_results.md"
    with open(output, "w") as f:
        print(f"Writing benchmark report to: {output}")
        f.write(md)
