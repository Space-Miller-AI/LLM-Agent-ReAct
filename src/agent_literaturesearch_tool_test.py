import ast
import os
import pickle
from pathlib import Path

import fire
from agents.top_down_agent import TopDownAgent
from findpapers_tool import FindPapersWrapper
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain.llms import OpenAI
from paperqa import Docs
from literature_search_tool import LiteratureSearchTool
from qa_tool import QAWrapper
from agents.findpapers_prompt import SEARCH_QUERY_PROMPT

def answer(
    query: str = "What is the aim of Optical Character Recognition (OCR)?",
    agent_type: str = "zero-shot-react-description",
    doc_store: str = None,
    max_iterations: int = 3,
    dig_deep: bool = False,
    nr_paper_per_db: dict = ast.literal_eval(os.getenv("NR_PAPER_PER_DB")),
) -> str:
    docs = Docs()
    
    if doc_store is not None and Path(doc_store).exists():
        with open(doc_store, "rb") as f:
            docs = pickle.load(f)

    llm = OpenAI(temperature=0.1, model_name='gpt-3.5-turbo')

    findpapers_api_wrapper = FindPapersWrapper(docs=docs, nr_paper_per_db=nr_paper_per_db, SEARCH_QUERY_PROMPT=SEARCH_QUERY_PROMPT)
    qa_api_wrapper = QAWrapper(docs=docs)
    tools = [LiteratureSearchTool(findpapers_api_wrapper=findpapers_api_wrapper, qa_api_wrapper=qa_api_wrapper)]
    agent = initialize_agent(
        tools, llm, agent=agent_type, verbose=True, max_iterations=max_iterations
    )

    if dig_deep:
        base_agent = agent

        def _run_agent(query):
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
            max_iterations=3,
        )

    r = agent.run(query)

    if doc_store is not None:
        with open(doc_store, "wb") as f:
            pickle.dump(docs, f)

    return r

if __name__ == "__main__":
    fire.Fire(answer)
