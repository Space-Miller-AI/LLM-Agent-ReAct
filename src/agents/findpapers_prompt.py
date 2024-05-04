# The main part of this prompt is copied from findpapers github repo (https://github.com/jonatasgrosman/findpapers) and modified to fit our needs.

SEARCH_QUERY_PROMPT = """
Generate a search query using only keywords that can be used to find relevant papers in different databases to answer the question.
Do not mention 'valid', 'query', 'search query', 'answer', 'research paper', 'pubmed', 'research', 'Abstract', 'arxiv' or similar terms in the search query.
The search query must not contain more than 4 keywords. The search query must not be complex. Join the keywords with ANDs or ORs.The search queries must follow these rules:
All the query terms need to be not empty and enclosed by square brackets. E.g., [term a]
The query can contain boolean operators, but they must be uppercase. The allowed operators are AND, OR, and NOT. E.g., [term a] AND [term b]
All the operators must have at least one whitespace before and after them (tabs or newlines can be valid too). E.g., [term a] OR [term b] OR [term c]
The NOT operator must always be preceded by an AND operator E.g., [term a] AND NOT [term b]
A subquery needs to be enclosed by parentheses. E.g., [term a] AND ([term b] OR [term c])
The composition of terms is only allowed through boolean operators. Queries like "[term a] [term b]" are invalid \n
We still have a few more rules that are only applicable on bioRxiv and medRxiv databases:
On subqueries with parentheses, only 1-level grouping is supported, i.e., queries with 2-level grouping like [term a] OR (([term b] OR [term c]) AND [term d]) are considered invalid
Only "OR" connectors are allowed between parentheses, i.e., queries like ([term a] OR [term b]) AND ([term c] OR [term d]) are considered invalid
Only "OR" and "AND" connectors are allowed, i.e., queries like [term a] AND NOT [term b] are considered invalid
Mixed connectors are not allowed on queries (or subqueries when parentheses are used), i.e., queries like [term a] OR [term b] AND [term b] are considered invalid. But queries like [term a] OR [term b] OR [term b] are considered valid \n
Let's see some examples of valid and invalid queries (YES means valid, and NO means invalid)):
| [term a]   |  Yes  |
| ([term a] OR [term b])   |  Yes  |
| [term a] OR [term b]  |  Yes  |
| [term a] AND [term b]   |  Yes  |
| [term a]OR[term b]   |  **No** (no whitespace between terms and boolean operator)  |
| ([term a] OR [term b]  |  **No** (missing parentheses)  |
| [term a] or [term b]  |  **No** (lowercase boolean operator)  |
| term a OR [term b]  |  **No** (missing square brackets)  |
| [term a] [term b]  |  **No** (missing boolean operator)  |
| [term a] XOR [term b] |  **No** (invalid boolean operator)   |
| [term a] OR NOT [term b] |  **No** (NOT boolean operator must be preceded by AND)   |
| [] AND [term b]  |  **No** (empty term)  |
|[some term*]  |  **No** (wildcards can be used only in single terms)  |
|[?erm]  |  **No** (wildcards cannot be used at the start of a search term)  |
|[te*]  |  **No** (a minimum of 3 characters preceding the asterisk wildcard is required)  |
|[ter*s]  |  **No** (the asterisk wildcard can only be used at the end of a search term)  |
|[t?rm?]  |  **No** (only one wildcard can be included in a search term)  |
The search query should be generated in the following JSON format: {{"search_query": ...}}
"""
