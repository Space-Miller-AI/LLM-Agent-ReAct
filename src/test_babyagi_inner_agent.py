# import the libraries from langchain used in this file
import ast
import os

from typing import Optional

from agents.babyagi_inner_agent import BabyAGI, intialize_vectorstore
from fire import Fire
from langchain.llms import OpenAI


def answer(
    objective: str = "Write a weather report for SF today",
    max_iterations: Optional[int] = 2,
    nr_paper_per_db: dict = ast.literal_eval(os.getenv("NR_PAPER_PER_DB")),
    doc_store: str = None,
):
    vectorstore = intialize_vectorstore()
    baby_agi = BabyAGI.from_llm(
        llm=OpenAI(temperature=0.1),
        vectorstore=vectorstore,
        verbose=False,
        max_iterations=max_iterations,
        nr_paper_per_db=nr_paper_per_db,
        doc_store=doc_store,
    )

    res = baby_agi({"objective": objective})
    print(res)


if __name__ == "__main__":
    Fire(answer)
