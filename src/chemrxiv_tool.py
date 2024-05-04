# Github Link of Chemrxiv API https://github.com/bingyinh/ChemRxiv_API/blob/main/chemrxiv_wrapper.py

import re

import requests
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from paperqa import Docs
from pubmed_tools import preprocess_authors
from pydantic import BaseModel
from utilities.chemrxiv_wrapper import chemrxiv_api, format_time


class ChemrxivWrapper(BaseModel):
    docs: Docs
    date_regex = r"\d{4}-\d{2}-\d{2}"

    class Config:
        arbitrary_types_allowed = True

    def _get_citation_chemrxiv(self, paper) -> str:
        """
        Generate the citation of a paper which includes a list of authors, paper's title and date.
        """

        try:
            authors_list = [
                f"{author_dict['lastName']}, {author_dict['firstName']}"
                for author_dict in paper["item"]["authors"]
            ]

            preprocessed_authors_list = preprocess_authors(authors_list)
            title = paper["item"]["title"]
            date = re.findall(self.date_regex, paper["item"]["publishedDate"])[0]
            citation = f"{preprocessed_authors_list}. {title}. {date}."
        except Exception as e:
            print(e)
            return None
        return citation

    def run(self, query: str) -> str:
        llm = OpenAI(temperature=0.2)
        template = """Here is a question:
            {query}
            Generate a search query that can be used to find relevant papers to answer the question.
            Do not mention 'research paper', 'pubmed', 'research', 'Abstract', 'Arxiv' or similar terms in the search query.
            Put quotation marks around relevant keywords and join them with ANDs.\n\n"""
        field_template = PromptTemplate(input_variables=["query"], template=template)
        field_chain = LLMChain(llm=llm, prompt=field_template)
        search_query = field_chain.run(query)
        search_query = search_query.replace('"', "")
        print(f"Generated Search Query : {search_query}")

        response = chemrxiv_api(
            term=search_query,
            skip=10,
            limit=15,
            sort="VIEWS_COUNT_DESC",
            searchDateFrom=format_time("2020/02/29"),
            searchDateTo=format_time("2023/03/01"),
        )

        results = response.json()["itemHits"]
        failed_abstracts = []
        return_msg = ""

        for result in results:
            paper_id = result["item"]["id"]
            file_full_path = f"../Documents/Chemrxiv/abstract_{paper_id}.txt"

            citation = self._get_citation_chemrxiv(result)

            if citation is None:
                print(f"Failed to extract citation for paper: {paper_id}")
                citation = f"Paper {paper_id}"

            return_msg += f"Found abstract: {citation}\n"
            if file_full_path in self.docs.docs:
                print("This abstract already in the Docs")
            else:
                print(f"Extracting and Indexing Abstract : {paper_id}")

                # extract the entire pdf
                pdf_url = result["item"]["asset"]["original"]["url"]
                response = requests.get(pdf_url)
                with open(file_full_path, "wb") as file:
                    file.write(response.content)

                # we can also extract only the abstract of the paper
                #                 abstract = result['item']['abstract']
                #                 with open(file_full_path, "w", encoding="utf-8") as file:
                #                     file.write(abstract)
                try:
                    self.docs.add(file_full_path, citation=citation)
                except Exception as e:
                    print(e, type(e))
                    print(f"Failed to index Abstract: {paper_id}")
                    failed_abstracts.append(paper_id)

        return return_msg


class ChemrxivTool(BaseTool):
    name = "ChemrxivTool"
    description = """ This tool is used to generate a search query for a given question, find relevant documents in Chemrxiv, download them and add to the library. The input is a natural language question."""
    api_wrapper: ChemrxivWrapper = None

    def __init__(self, docs: Docs):
        api_wrapper = ChemrxivWrapper(docs=docs)
        super().__init__(api_wrapper=api_wrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        print("\n")
        res_bool = self.api_wrapper.run(query)
        return res_bool

    async def _arun(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError
