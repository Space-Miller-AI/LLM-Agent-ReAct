# LLM-Agent-ReAct

This project focuses on developing an advanced custom AI agent based on ReAct logic aimed at generating factual answers with references based on research publications. Language models are getting better at reasoning (e.g. chain-of-thought prompting) and acting (e.g. WebGPT, SayCan, ACT-1). ReAct integrates these two effective methods—reasoning and acting—into a unified approach, enhancing the overall capabilities of language models.

Key features of the Project:
* Agent performs multi-step self-questioning and answering until the agent thinks it can answer the original question. It consist of a tool which helps the agent answer sub-questions.
* For each sub-question, the agent's tool extracts relevant research papers across various databases, including arXiv, ChemRxiv, PubMed, and others through API requests.
* The tool performs semantic search on extracted papers to find the most relevant information, which is used as context to enable the GPT-3.5-turbo model to generate highly accurate and factual responses.
* The system generates as final output the answer and the reference paper based on which the answers was generated.
* Outputs from the language model are in JSON, simplifying the extraction of necessary details like answers and references, thus enhancing usability and integration capability.
