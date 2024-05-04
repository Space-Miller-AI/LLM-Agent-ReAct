import re

import findpapers
import findpapers.utils.persistence_util as persistence_util
import requests
from agents.findpapers_prompt import SEARCH_QUERY_PROMPT
from bs4 import BeautifulSoup
from findpapers.models.paper import Paper
from typing import Union
from langchain.tools.base import BaseTool
from langchain.utilities import BingSearchAPIWrapper
from load_env_vars import (
    BING_SEARCH_KEY,
    BING_SUB_KEY,
    OUTPUT_DIRECTORY,
    SEARCH_DIRECTORY,
)
from paperqa import Docs
from base_tool import BaseWrapper


class FindPapersWrapper(BaseWrapper):
    """
    Wrapper for all papers tool which is used to seach, download and index papers from all sources
    """

    nr_paper_per_db: dict

    def _generate_citation(self, paper: Union[Paper, dict], file_full_path: str) -> str:
        """
        Generate the citation of a FindPapers Paper.

        Args:
            paper (Paper): Paper from FindPapers
            file_full_path (str): Full path of the paper
        Returns:
            citation (str): Generated citation of the paper
        """

        try:
            if isinstance(paper, Paper):
                citation = f"{self._preprocess_authors(paper.authors)}. {paper.title}. {paper.publication_date.strftime('%Y-%m-%d')}."
            else:
                paper_article_url = (
                    "https://chemrxiv.org/engage/chemrxiv/article-details/" + paper
                )
                response_article = requests.get(paper_article_url)
                soup = BeautifulSoup(response_article.content, "html.parser")

                authors = [
                    el.get("content")
                    for el in soup.find_all("meta", attrs={"name": "citation_author"})
                ]
                title = soup.find_all("meta", attrs={"name": "citation_title"})[0].get(
                    "content"
                )
                attr_val = "v-div d-inline-block primary--text font-weight-bold"
                date = soup.find_all("div", attrs={"class": attr_val})[0].text.split(
                    "Version"
                )[0]
                citation = f"{self._preprocess_authors(authors)}. {title}. {date}."
        except Exception:
            print(
                f"Failed to generate citation for Paper {file_full_path}. Process with citation generation using LLM."
            )
            citation = self._generate_citation_llm(file_full_path)

        return citation


    def run(self, query: str) -> str:
        """
        Run the all papers tool to search, download and index papers from all sources.

        Args:
            query (str): Query to search for papers

        Returns:
            return_message (str): Message to be returned
        """

        search_query = self._generate_search_query(query)

        # use the code te below to test the case where no papers are downloaded. Uncomment the code below and set n_iter to 0 as instance variable for this class
        # if self.nr_iter == 0:
        #     search_query = '[ssdgdsgdgsdgsgodsnmog]'
        # self.nr_iter += 1

        print(f"Generated Search Query : {search_query}")

        # OUTPUT_DIRECTORY is the directory where the papers will be downloaded
        # SEARCH_DIRECTORY is the directory where the search results (paper's metadata) will be saved
        characters_to_replace = [
            " ",
            "_",
            "]",
            "[",
            ")",
            "(",
            "*",
        ]  # List of characters to replace
        
        processed_search_query = search_query
        for char in characters_to_replace:
            processed_search_query = processed_search_query.replace(char, "_")
        
        selected_findpapers_databases = [db for db in self.nr_paper_per_db if db != "chemrxiv" and self.nr_paper_per_db[db] > 0]
        searchpath_db_json_list = [
            SEARCH_DIRECTORY + f"/{processed_search_query}_{database}_search.json"
            for database in selected_findpapers_databases
        ]

        start_search_index = 1
        try:
            findpapers.search(
                outputpath=searchpath_db_json_list[0],
                query=search_query,
                limit=self.nr_paper_per_db[selected_findpapers_databases[0]],
                databases=[selected_findpapers_databases[0]],
                limit_per_database=self.nr_paper_per_db[selected_findpapers_databases[0]],
            )

        except ValueError:
            print(f"This Search Query : {search_query} format is not valid!")
            # get only the first keyword if the generated search query is not valid
            first_idx = search_query.index("[")
            second_idx = search_query.index("]")
            search_query = search_query[first_idx : second_idx + 1]
            start_search_index = 0
            print(f"Proceeding with the following search query: {search_query}")

        for database, searchpath_db_json in zip(
            selected_findpapers_databases[start_search_index:],
            searchpath_db_json_list[start_search_index:],
        ):
            try:
                findpapers.search(
                    outputpath=searchpath_db_json,
                    query=search_query,
                    limit=self.nr_paper_per_db[database],
                    databases=[database],
                    limit_per_database=self.nr_paper_per_db[database],
                )

            except Exception as e:
                print(
                    f"Papers Search for the database {database} Failed! Error Message: {e}"
                )

        # download the papers that were extracted using findpapers.search locally
        # use findpapers.download to download the papers (only PubMed abstracts are not downloaded)
        for database, searchpath_db_json in zip(
            selected_findpapers_databases, searchpath_db_json_list
        ):
            try:
                findpapers.download(
                    search_path=searchpath_db_json,
                    output_directory=OUTPUT_DIRECTORY,
                )
            except Exception as e:
                print(
                    f"Paper downloading for the database {database} failed! Error Message: {e}"
                )

        # get the best resulting papers from FindPapers for the given search query
        findpapers_results = []
        for searchpath_db_json in searchpath_db_json_list:
            # database_name = searchpath_db_json.split("_")[-2]
            findpapers_results += list(persistence_util.load(searchpath_db_json).papers)

        # get the best resulting papers from Chemrxiv for given search query
        if "chemrxiv" in self.nr_paper_per_db and self.nr_paper_per_db['chemrxiv'] > 0:
                search = BingSearchAPIWrapper(
                    k=self.nr_paper_per_db["chemrxiv"],
                    bing_subscription_key=BING_SUB_KEY,
                    bing_search_url=BING_SEARCH_KEY,
                )

                try:
                    chemrxiv_results = [
                        paper
                        for paper in search.results(
                            f"site:chemrxiv.org {search_query} .pdf",
                            self.nr_paper_per_db["chemrxiv"],
                        )
                        if paper["link"].endswith(".pdf")
                    ]

                except Exception as e:
                    chemrxiv_results = []
                    print(
                        f"Could not get any search results from Chemrxiv! Error Message: {e}"
                    )

        else:
            chemrxiv_results = []

        # Concatenate FindPapers and Chemrxiv papers together
        all_results = findpapers_results + chemrxiv_results
        return_msg = ""

        
        all_citations_docstore = [
            doc_metadata[2] for doc_metadata in self.docs.doc_previews()
        ]

        if len(all_results) > 0:
            for paper in all_results:
                # check if the paper is coming from FindPapers or Chemrxiv (Paper is a datatype of FindPapers package)
                if isinstance(paper, Paper):
                    # Case FindPapers Paper
                    paper_id = re.sub(
                        r"[^\w\d-]", "_", f"{paper.publication_date.year}-{paper.title}"
                    )
                    file_full_path_no_extension = OUTPUT_DIRECTORY + f"/{paper_id}"

                    # For some papers (like PubMed) only abstracts are available. They do not have pdf urls.
                    if paper.urls == set():
                        file_full_path = file_full_path_no_extension + ".txt"
                        
                        try:
                            abstract = paper.abstract

                            with open(file_full_path, "w", encoding="utf-8") as file:
                                file.write(abstract)
                        except Exception as e:
                            print(
                                f"Downloading PubMed Paper Failed. File Path : {file_full_path}. Error Message: {e}"
                            )
                            continue

                    else:
                        file_full_path = file_full_path_no_extension + ".pdf"
                    
                    citation = self._generate_citation(paper, file_full_path)
                    
                else:
                    # Case Chemrxiv Paper
                    paper_pdf_url = paper["link"]
                    paper_id = paper_pdf_url.split("item/")[-1].split("/original")[0]
                    file_full_path = OUTPUT_DIRECTORY + f"/{paper_id}.pdf"

                    response_pdf = requests.get(paper_pdf_url)
                    with open(file_full_path, "wb") as file:
                        file.write(response_pdf.content)

                    citation = self._generate_citation(
                        paper_id, file_full_path
                    )

                if citation in all_citations_docstore:
                        print(
                            f"Paper {file_full_path} already in collection. Skipping..."
                        )
                        return_msg += f"Found but did not Index Chemrxiv Paper: {file_full_path} since it is already in collection.\n"
                else: 
                    return_msg = self._index_paper(
                        file_full_path, citation, return_msg,
                    )
                
        else:
            return_msg = f"No papers found for this search query {search_query} !"
        return return_msg

class FindPapersTool(BaseTool):
    """
    This tool is used to generate a search query for a given question, find relevant papers
    in the web, download them and add to the library. The input is a natural language question.
    """

    name = "AllPapersTool"
    description = """ This tool is used to generate a search query for a given question, find relevant papers in the web, download them and add to the library. The input is a natural language question."""
    api_wrapper: FindPapersWrapper

    def __init__(self, docs: Docs, nr_paper_per_db: dict):
        api_wrapper = FindPapersWrapper(docs=docs, nr_paper_per_db=nr_paper_per_db, SEARCH_QUERY_PROMPT=SEARCH_QUERY_PROMPT)
        super().__init__(api_wrapper=api_wrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        print("\n")
        res_bool = self.api_wrapper.run(query)
        return res_bool

    async def _arun(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError
