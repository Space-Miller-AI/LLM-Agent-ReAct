# import the libraries from langchain used in this file
from typing import Optional

from agents.babyagi_outer_agent import BabyAGI, intialize_vectorstore
from fire import Fire
from langchain.llms import OpenAI


def answer(
    objective: str = "Write a weather report for SF today",
    max_iterations: Optional[int] = 3,
    nr_paper_per_db: dict = {
        "scopus": 2,
        "biorxiv": 2,
        "acm": 2,
        "arxiv": 2,
        "pubmed": 2,
        "medrxiv": 2,
        "chemrxiv": 2,
    },
):
    vectorstore = intialize_vectorstore()
    baby_agi = BabyAGI.from_llm(
        nr_paper_per_db=nr_paper_per_db,
        llm=OpenAI(temperature=0),
        vectorstore=vectorstore,
        verbose=False,
        max_iterations=max_iterations,
    )

    res = baby_agi({"objective": objective})
    print(res)


if __name__ == "__main__":
    Fire(answer)
