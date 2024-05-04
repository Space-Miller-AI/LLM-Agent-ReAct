import pickle
from pathlib import Path

import fire
from agents.top_down_agent import TopDownAgent
from chemrxiv_tool import ChemrxivTool
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain.llms import OpenAI
from paperqa import Docs
from pdf_tool import QATool


def answer(
    query: str,
    agent_type="zero-shot-react-description",
    dig_deep=False,
    doc_store: str = None,
    max_iterations=5,
) -> str:
    docs = Docs()
    if doc_store is not None and Path(doc_store).exists():
        with open(doc_store, "rb") as f:
            docs = pickle.load(f)

    llm = OpenAI(temperature=0.2)
    tools = [ChemrxivTool(docs), QATool(docs)]
    agent = initialize_agent(
        tools, llm, agent=agent_type, verbose=True, max_iterations=max_iterations
    )
    if dig_deep:
        base_agent = agent

        def _run_agent(query):
            # print(f"Running agent on query:{query}")
            return base_agent.run(query)

        search_tool = Tool(
            name="Research",
            description="Answers a question by researching relevant literatur",
            func=_run_agent,
        )
        agent_obj = TopDownAgent.from_llm_and_tools(llm, [search_tool], verbose=True)
        agent = AgentExecutor.from_agent_and_tools(
            agent_obj,
            [search_tool],
            verbose=True,
            early_stopping_method="generate",
            max_iterations=max_iterations,
        )

    r = agent.run(query)

    if doc_store is not None:
        with open(doc_store, "wb") as f:
            pickle.dump(docs, f)

    return r


if __name__ == "__main__":
    fire.Fire(answer)
