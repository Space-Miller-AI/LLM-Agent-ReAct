# to run this file you need the latest langchain version (not the one in toml file)

import pickle
from pathlib import Path

from all_papers_tool import AllPapersTool
from fire import Fire
from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import (
    load_agent_executor,
    load_chat_planner,
    PlanAndExecute,
)
from paperqa import Docs
from pdf_tool import QATool


def answer(
    query: str,
    doc_store: str = None,
    nr_paper_per_db: dict = {
        "scopus": 7,
        "biorxiv": 7,
        "acm": 7,
        "arxiv": 7,
        "pubmed": 30,
        "medrxiv": 7,
        "chemrxiv": 7,
        "ieee": 7,
    },
):
    docs = Docs()
    if doc_store is not None and Path(doc_store).exists():
        with open(doc_store, "rb") as f:
            docs = pickle.load(f)

    tools = [AllPapersTool(docs, nr_paper_per_db=nr_paper_per_db), QATool(docs)]
    model = ChatOpenAI(temperature=0.1)
    planner = load_chat_planner(model)
    executor = load_agent_executor(model, tools, verbose=True)
    agent = PlanAndExecute(
        planner=planner, executor=executor, verbose=True, max_iterations=3
    )
    answer = agent.run(query)
    print(answer)


if __name__ == "__main__":
    Fire(answer)
