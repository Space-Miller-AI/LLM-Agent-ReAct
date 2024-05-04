from unpywall.utils import UnpywallCredentials

UnpywallCredentials('lorenczhula28@gmail.com')

from paperqa import Docs
from unpywall import Unpywall
from langchain.tools.base import BaseTool
from base_tool import BaseWrapper
from typing import Union


class UnpywallWrapper(BaseWrapper):
    """
    Wrapper for all papers tool which is used to seach, download and index papers from all sources
    """

    nr_papers: int

    def _generate_citation(self, paper, file_full_path: str) -> str:
        """
        Generate the citation for a paper.

        Args:
            paper (Union[Paper, dict]): Paper object or paper metadata.
            file_full_path (str): Full path of the paper.

        Returns:
            citation (str): Generated citation of the paper.
        """
        try:
            authors = [f'{author["family"]} {author["given"]}' for author in paper['z_authors']] if paper['z_authors'] else 'No Authors Available'
            title = paper['title'] if paper['title'] else 'No Title Available'
            date = paper['published_date'] if paper['published_date'] else "No Date Available"
            citation = f'{self._preprocess_authors(authors)}. {title}. {date}'

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
        print(f"Generated Search Query : {search_query}")

        res = Unpywall.query(query=search_query, is_oa=True)
        
        if res.shape[0] > 0:
            res = res.sample(frac=1, random_state=42)[:self.nr_papers]

            file_path = 'data'
            return_msg = ''
            for _, row in res.iterrows():
                if row['doi'] and row['best_oa_location.url_for_pdf']:
                    if row['doi'] not in self.docs.keys:
                        file_name = f"{row['doi'].replace('/', '')}.pdf"
                        citation = self._generate_citation(row, file_name)
                        Unpywall.download_pdf_file(row['doi'], file_name, file_path)

                        return_msg = self._index_paper(file_path + f'/{file_name}', citation, return_msg, row['doi'])
                        # self.docs.add(path=file_path + f'/{file_name}', citation=citation, disable_check=True, key=row['doi'])
                    else:
                        print(
                                f"Paper {file_name} already in collection. Skipping..."
                            )
                        return_msg += f"Found but did not Index Paper: {file_name} since it is already in collection.\n"

        else:
            return_msg = f"No papers found for this search query {search_query} !"

        return return_msg 
    
class UnpywallTool(BaseTool):
    """
    This tool is used to generate a search query for a given question, find relevant papers
    in the web, download them and add to the library. The input is a natural language question.
    """

    name = "AllPapersTool"
    description = """ This tool is used to generate a search query for a given question, find relevant papers in the web, download them and add to the library. The input is a natural language question."""
    api_wrapper: UnpywallWrapper

    def __init__(self, docs: Docs, nr_paper_per_db: dict):
        api_wrapper = UnpywallWrapper(docs=docs, nr_paper_per_db=nr_paper_per_db)
        super().__init__(api_wrapper=api_wrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        print("\n")
        res_bool = self.api_wrapper.run(query)
        return res_bool

    async def _arun(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError