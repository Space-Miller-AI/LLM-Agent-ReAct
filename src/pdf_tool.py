import download_pdf
from langchain.tools.base import BaseTool
from paperqa import Docs
from pydantic import BaseModel


class PdfDownloadWrapper(BaseModel):
    method = "pypdf"
    docs: Docs

    class Config:
        arbitrary_types_allowed = True

    def run(self, url: str) -> str:
        url = download_pdf.sanitize_url(url)
        filename = download_pdf.get_filename_from_url(url)
        if filename in self.docs.docs:
            print(f"File {filename} already exists. Skipping download.")
        else:
            filename = download_pdf.download_pdf(url)
            try:
                self.docs.add(filename)
            except ValueError:
                pass
            except Exception as e:
                print(e, type(e))
                print(f"Failed to add {filename} to docs.")

        output = f"Downloaded {filename} and added to docs."
        # data = load_pdf.load_pdf("../temp.pdf")
        # output = "\n\n".join([p.page_content for p in data])
        return output


class PdfDownloadTool(BaseTool):
    name = "Download PDFs"
    description = """This tool is useful for downloading PDFs from arXiv and extracting the text of it.
                  Input should be a list of URLs to PDF separated by newline."""
    api_wrapper: PdfDownloadWrapper

    def __init__(self, docs: Docs):
        api_wrapper = PdfDownloadWrapper(docs=docs)
        super().__init__(api_wrapper=api_wrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        temp = ""
        print("\n")
        for paper in query.split("\n"):
            print(f"Downloading and indexing paper: {paper}")
            try:
                temp = temp + self.api_wrapper.run(paper) + "\n"
            except Exception as e:
                print(e)
        return temp

    async def _arun(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError
