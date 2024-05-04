from abc import ABC, abstractmethod
from typing import Union
from pydantic import BaseModel
from paperqa import Docs 
from findpapers.models.paper import Paper
from paperqa.qaprompts import citation_prompt, make_chain
from paperqa.readers import read_doc
import os
from datetime import datetime
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import json 

class BaseWrapper(BaseModel):
    """
    Wrapper for all papers tool which is used to seach, download and index papers from all sources
    """

    docs: Docs
    SEARCH_QUERY_PROMPT: str

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def _generate_citation(self, paper: Union[Paper, dict], file_full_path: str) -> str:
        """
        Generate the citation for a paper.

        Args:
            paper (Union[Paper, dict]): Paper object or paper metadata.
            file_full_path (str): Full path of the paper.

        Returns:
            citation (str): Generated citation of the paper.
        """
        pass

    @abstractmethod
    def run(self, query: str) -> str:
        """
        Process a query and return the result.

        Args:
            query (str): Query to be processed.

        Returns:
            result (str): Result of processing the query.
        """
        pass

    def _preprocess_authors(self, authors_list: list) -> str:
        """
        Preprocess the list of authors to be used in citation.
        If there are more than 2 authors, only the first author will be mentioned followed by 'et al'.

        Args:
            authors_list (list): List of authors

        Returns:
            authors_list (str): Preprocessed authors list
        """

        if len(authors_list) > 2:
            authors_list = f"{authors_list[0]}, et al"
        else:
            authors_list = ", and ".join(authors_list)

        return authors_list
    

    def _index_paper(
        self, file_full_path: str, citation: str, return_msg: str, key: str = None
    ) -> str:
        """
        Index a paper to the vector database.

        Args:
            file_full_path (str): Full path of the paper
            citation (str): Citation of the paper
            return_msg (str): current message to be returned
        Returns:
            return_msg (str): updated message to be returned
        """

        try:
            self.docs.add(file_full_path, citation=citation, key=key)
            return_msg += f"Found and Indexed Paper: {citation}\n"
        except ValueError as e:
            print(f"Did not index Paper: Error Message: {e}")
            return_msg += f"Found but did not Index Paper: {citation} since it is already in collection.\n"
        except Exception as e:
            print(f"Failed to index Paper: {citation}. Error Message: {e}")
            return_msg += f"Found but could not Index  Paper: {citation} since it could not be downloaded.\n"

        return return_msg
    

    def _generate_citation_llm(self, file_full_path: str) -> str:
        """
        Generate the citation of a Paper using LLM if the citation cannot be generated using the functions _generate_citation_findpapers and _generate_citation_chemrxiv.

        Args:
            file_full_path (str): Full path of the paper
        Returns:
            citation (str): Generated citation of the paper using LLM
        """
        try:
            cite_chain = make_chain(prompt=citation_prompt, llm=self.docs.summary_llm)
            texts, _ = read_doc(
                file_full_path, "", "", chunk_chars=self.docs.chunk_size_limit
            )
            citation = cite_chain.run(texts[0])

        except FileNotFoundError:
            print(f"Paper {file_full_path} is not downloaded.")
            citation = None
        except Exception as e:
            print(
                f"Could not generate citation for Paper {file_full_path}. Error Message: {e}"
            )
            citation = None

        if citation is None:
            citation = f"No Citation, {os.path.basename(file_full_path)}, {datetime.now().year}"
        else:
            if (
                len(citation) < 3
                or "Unknown" in citation
                or "insufficient" in citation
                or "not provide" in citation
            ):
                citation = f"Unknown, {os.path.basename(file_full_path)}, {datetime.now().year}"
        return citation
    

    def _generate_search_query(self, query: str) -> str:
        """
        Generate and format the search query to search on FindPapers and Chemrxiv databases for a given query using LLM.

        Args:
            query (str): Query to search for
        Returns:
            search_query (str): Generated search query
        """
        llm = OpenAI(temperature=0.2, model_name='gpt-3.5-turbo')
        template = "Here is a question:\n{query}\n" + self.SEARCH_QUERY_PROMPT + "\n\n"
        field_template = PromptTemplate(input_variables=["query"], template=template)
        field_chain = LLMChain(llm=llm, prompt=field_template)
        response_search_query = field_chain.run(query)
        json_search_query = json.loads(response_search_query)
        return json_search_query['search_query']