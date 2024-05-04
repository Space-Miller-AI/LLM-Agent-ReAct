"""Attempt to implement MRKL systems as described in arxiv.org/pdf/2205.00445.pdf."""
from __future__ import annotations

# import re
from typing import Any, List, Optional, Sequence, Union

from langchain.agents.agent import Agent, AgentOutputParser
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.base import BaseTool
from pydantic import Field

from .top_down_prompts import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX

FINAL_ANSWER_ACTION = "Final Answer:"


class TopDownOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        sub_question = text.split("Sub-question:")[-1].strip()
        return AgentAction("Research", sub_question.strip(" ").strip('"'), text)


class TopDownAgent(Agent):
    """Agent that patiently ponders about a question."""

    output_parser: AgentOutputParser = Field(default_factory=TopDownOutputParser)

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "top-down-scholar"

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Assistant answer: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return TopDownOutputParser()

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.
        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            input_variables: List of input variables the final prompt will expect.
        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        format_instructions = format_instructions.format()
        template = "\n\n".join([prefix, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLLM,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        """Validate that the tools are valid for this agent."""
        for tool in tools:
            if tool.name != "Research":
                raise ValueError(
                    f"Patient Scholar Agent can only use Research tool, but got {tool.name}"
                )
        if len(tools) != 1:
            raise ValueError(
                f"Patient Scholar Agent can only use one tool, but got {len(tools)}"
            )
