# import the necessary packages used in this file from langchain
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, ZeroShotAgent
from paperqa import Docs
from all_papers_tool import AllPapersTool
from pdf_tool import QATool

def main(query='What are the advantages and disadvantages of batch normalization compared to other normalization techniques?'):

    prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
    {agent_scratchpad}"""
    
    nr_paper_per_db: dict = {
        "scopus": 5,
        "biorxiv": 6,
        "acm": 7,
        "arxiv": 4,
        "pubmed": 10,
        "medrxiv": 9,
        "chemrxiv": 3,
        }
    docs = Docs()
    llm = OpenAI(temperature=0.2)
    tools = [AllPapersTool(docs, nr_paper_per_db=nr_paper_per_db), QATool(docs)]
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["objective", "task", "context", "agent_scratchpad"],
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    agent_executor.run(query)

if __name__ == '__main__':
    main()
    