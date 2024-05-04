from pydantic import BaseModel
from findpapers_tool import FindPapersWrapper
from qa_tool import QAWrapper
from langchain.tools.base import BaseTool

class LiteratureSearchWrapper(BaseModel):
    """
    Wrapper for all papers tool which is used to seach, download and index papers from all sources
    """

    findpapers_api_wrapper: FindPapersWrapper
    qa_api_wrapper: QAWrapper

    class Config:
        arbitrary_types_allowed = True

    def run(self, query: str) -> str:
        self.findpapers_api_wrapper.run(query)
        return self.qa_api_wrapper.run(query)

class LiteratureSearchTool(BaseTool):
    """
    This tool is used to generate a search query for a given question, find relevant papers
    in the web, download them and add to the library. The input is a natural language question.
    """

    name = "LiteratureSearchTool"
    description = """ This tool is used for answering questions."""
    literaturesearch_api_wrapper: LiteratureSearchWrapper

    def __init__(self, findpapers_api_wrapper: FindPapersWrapper, qa_api_wrapper: QAWrapper):
        super().__init__(literaturesearch_api_wrapper=LiteratureSearchWrapper(findpapers_api_wrapper=findpapers_api_wrapper, qa_api_wrapper=qa_api_wrapper))

    def _run(self, query: str) -> str:
        """Use the tool."""
        print("\n")
        return self.literaturesearch_api_wrapper.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError