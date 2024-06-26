# import the libraries from langchain used in this file
from collections import deque
from typing import Any, Dict, List, Optional
from agents.top_down_agent import TopDownAgent
from langchain.agents import initialize_agent

from all_papers_tool import AllPapersTool


from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents import AgentExecutor, Tool, SelfAskWithSearchAgent
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from paperqa import Docs
from pdf_tool import QATool
from pydantic import BaseModel, Field
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
import faiss


def intialize_vectorstore() -> VectorStore:
    """
    Initialize the FAISS vectorstore.
    """
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    return vectorstore


class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "result",
                "task_description",
                "incomplete_tasks",
                "objective",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


def get_next_task(
    task_creation_chain: LLMChain,
    result: Dict,
    task_description: str,
    task_list: List[str],
    objective: str,
) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(
        result=result,
        task_description=task_description,
        incomplete_tasks=incomplete_tasks,
        objective=objective,
    )
    new_tasks = response.split("\n")
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


def prioritize_tasks(
    task_prioritization_chain: LLMChain,
    this_task_id: int,
    task_list: List[Dict],
    objective: str,
) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(
        task_names=task_names, next_task_id=next_task_id, objective=objective
    )
    new_tasks = response.split("\n")
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list


def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata["task"]) for item in sorted_results]


def execute_task(
    vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5
) -> str:
    """Execute a task."""
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execution_chain.run(objective=objective, context=context, task=task)

class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if num_iters == 0:
                todo_prompt = PromptTemplate.from_template(
                    "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
                )
                todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
                excecuter = todo_chain
                task_results = ""
            else:
                excecuter = self.execution_chain

            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, excecuter, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                self.print_task_result(result)
                # task_results += result + "\n"
                task_results += (
                    f"TASK: {this_task_id}) {task['task_name']}:\n{result}\n\n"
                )

                # Step 3: Store the result in Pinecone
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain,
                    result,
                    task["task_name"],
                    [t["task_name"] for t in self.task_list],
                    objective,
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain,
                        this_task_id,
                        list(self.task_list),
                        objective,
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print(
                    "\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m"
                )
                break

        print(f"Task Results: \n{task_results}\n")
        return {}

    @classmethod
    def from_llm(
        cls, nr_paper_per_db: dict, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )

        docs = Docs()
        llm = OpenAI(temperature=0.2)
        tools = [AllPapersTool(docs, nr_paper_per_db=nr_paper_per_db), QATool(docs)]
        inner_agent_executer = initialize_agent(
            tools, llm, agent="zero-shot-react-description", verbose=True, max_iterations=4
        )
        base_inner_agent_executer = inner_agent_executer

        def _run_agent(query):
            # print(f"Running agent on query:{query}")
            return base_inner_agent_executer.run(query)

        search_tool = Tool(
            name="Research",
            description="Answers a question by researching relevant literatur",
            func=_run_agent,
        )
        outer_agent = TopDownAgent.from_llm_and_tools(llm, [search_tool], verbose=True)
        outer_agent_executer = AgentExecutor.from_agent_and_tools(
            outer_agent,
            [search_tool],
            verbose=True,
            early_stopping_method="generate",
            max_iterations=3,
        )

        base_outer_agent_executer = outer_agent_executer
        def _run_outer_agent(query):
            # print(f"Running agent on query:{query}")
            return base_outer_agent_executer.run(query)

        new_tools = [Tool(
            name="Research Papers and Answer Questions",
            description="Research for relevant papers and answer questions based on relevant papers.",
            func=_run_outer_agent,
        )]

        template = "You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\nQuestion: {task}\n{agent_scratchpad}"
        prompt = PromptTemplate(
            input_variables=["objective", "task", "context", "agent_scratchpad"], template=template
            )

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in new_tools]
        main_agent = SelfAskWithSearchAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        main_agent_executor = AgentExecutor.from_agent_and_tools(
            agent=main_agent, tools=new_tools, verbose=True
        )

        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=main_agent_executor,
            vectorstore=vectorstore,
            **kwargs,
        )
