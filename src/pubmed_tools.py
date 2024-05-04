from Bio import Entrez
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from paperqa import Docs
from pydantic import BaseModel


def preprocess_authors(authors_list: list) -> str:
    if len(authors_list) > 2:
        # et.al
        authors_list = f"{authors_list[0]}, et al"
    else:
        authors_list = ", and ".join(authors_list)

    return authors_list


def get_citation_pubmed(paper) -> str:
    # Generate citation in MLA format
    try:
        authors = paper["MedlineCitation"]["Article"]["AuthorList"]
        authors_list = []
        for author in authors:
            authors_list.append(f"{author['LastName']}, {author['ForeName']}")

        preprocessed_authors_list = preprocess_authors(authors_list)
        title = paper["MedlineCitation"]["Article"]["ArticleTitle"]
        journal = paper["MedlineCitation"]["Article"]["Journal"]["Title"]
        year = paper["MedlineCitation"]["Article"]["Journal"]["JournalIssue"][
            "PubDate"
        ]["Year"]
        citation = f"{preprocessed_authors_list}. {title}. {journal}. {year}."
    except KeyError:
        return None

    return citation


def get_abstract(paper) -> str:
    try:
        full_abstract = paper["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
    except KeyError:
        return ""

    if len(full_abstract) == 1:
        preprocessed_abstract = full_abstract[0]
    elif len(full_abstract) > 1:
        preprocessed_abstract = ""
        for sub_abs in full_abstract:
            sub_text = sub_abs.strip()
            sub_label = sub_abs.attributes["Label"]
            preprocessed_abstract += sub_label + ": " + sub_text + "\n"
    else:
        preprocessed_abstract = ""
    return preprocessed_abstract


class PubmedWrapper(BaseModel):
    docs: Docs

    class Config:
        arbitrary_types_allowed = True

    def run(self, query: str) -> str:
        llm = OpenAI(temperature=0.2)
        template = """Here is a question:
            {query}
            Generate a search query that can be used to find relevant documents in PubMed to answer the question.
            Do not mention 'research paper', 'pubmed', 'research', 'Abstract' or similar terms in the query.
            Put quotation marks around relevant keywords and join them with ANDs.\n\n"""
        field_template = PromptTemplate(input_variables=["query"], template=template)
        field_chain = LLMChain(llm=llm, prompt=field_template)
        search_query = field_chain.run(query)
        print(f"Generated Search Query : {search_query}")

        Entrez.email = "example@gmail.com"
        handle = Entrez.esearch(
            db="pubmed", retmax="5", retmode="xml", term=search_query
        )
        results = Entrez.read(handle)
        id_list = results["IdList"]
        if len(id_list) == 0:
            return "No papers found. Please try a different question."
        ids = ",".join(id_list)
        handle = Entrez.efetch(db="pubmed", retmode="xml", id=ids)
        papers = Entrez.read(handle)
        failed_abstracts = []
        return_msg = ""

        for paper_id, paper in zip(id_list, papers["PubmedArticle"]):
            abstract = get_abstract(paper)
            citation = get_citation_pubmed(paper)
            if citation is None:
                print(f"Failed to extract citation for paper: {paper_id}")
                citation = f"Paper {paper_id}"
            if len(abstract) == 0:
                print(f"Failed to extract abstract for paper: {paper_id}")
                failed_abstracts.append(paper_id)
                continue
            file_full_path = f"../Documents/PubMed_Abstracts/abstract_{paper_id}.txt"
            return_msg += f"Found abstract: {citation}\n"
            if file_full_path in self.docs.docs:
                print("This abstract already in the Docs")
            else:
                print(f"Extracting and Indexing Abstract : {paper_id}")

                with open(file_full_path, "w", encoding="utf-8") as file:
                    file.write(abstract)

                try:
                    self.docs.add(file_full_path, citation=citation)
                except Exception as e:
                    print(e, type(e))
                    print(f"Failed to index Abstract: {paper_id}")
                    failed_abstracts.append(paper_id)

        return return_msg


class PubmedTool(BaseTool):
    name = "PubmedTool"
    description = """ This tool is used to generate a search query for a given question, find relevant documents in PubMed, extract them and add to the library. The input is a natural language question."""
    api_wrapper: PubmedWrapper

    def __init__(self, docs: Docs):
        api_wrapper = PubmedWrapper(docs=docs)
        super().__init__(api_wrapper=api_wrapper)

    def _run(self, query: str) -> str:
        """Use the tool."""
        print("\n")
        res_bool = self.api_wrapper.run(query)
        return res_bool

    async def _arun(self, query: str) -> str:
        """Use the tool."""
        raise NotImplementedError
