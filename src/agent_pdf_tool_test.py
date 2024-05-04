import pickle
from pathlib import Path
from typing import Any

import fire
from agents.top_down_agent import TopDownAgent
from all_papers_tool import AllPapersTool
from chemrxiv_tool import ChemrxivTool
from langchain.agents import initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import OpenAI

# from langchain.chat_models import ChatOpenAI
from paperqa import Docs
from pdf_tool import QATool

# Get list of sample arxiv links


class SaveDocs(BaseCallbackHandler):
    def __init__(self, docs: Docs, filename: str):
        self.docs = docs
        self.filename = filename

    def on_agent_finish(self, finish, **kwargs: Any) -> Any:
        with open(self.filename, "wb") as f:
            pickle.dump(self.docs, f)


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
    # tools = [QATool(docs), PdfDownloadTool(docs), arxivtool]

    tools = [AllPapersTool(docs), ChemrxivTool(docs), QATool(docs)]
    llm = OpenAI(temperature=0.3)
    agent = initialize_agent(
        tools, llm=llm, agent=agent_type, verbose=True, max_iterations=max_iterations
    )

    callbacks = []
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
        callbacks.append(SaveDocs(docs, doc_store))

    r = agent.run(query, callbacks=callbacks)

    if doc_store is not None:
        with open(doc_store, "wb") as f:
            pickle.dump(docs, f)
    return r


if __name__ == "__main__":
    fire.Fire(answer)
