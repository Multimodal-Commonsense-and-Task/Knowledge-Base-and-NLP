
def get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_math_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_code_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Find the minimum number of vertices in a Steiner tree that includes all specified vertices in a given tree.\"\n"
        "Assistant thinking steps:\n"
        "- I need to understand what a Steiner tree is and how to compute the minimum number of vertices required to include all specified vertices in a given tree.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Minimum Steiner Tree problem in trees<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""

def get_llm_reorder_instruction(search_query, document, top_k=1):
    document = [f"[{i}]: {doc}" for i, doc in enumerate(document)]
    document = "\n\n".join(document)
    return (
        "You are a reranking assistant. "
        f"Sort the documents by their relevance to the query and return top-{top_k} documents\n"
        f"IMPORTANT: You SHOULD ONLY contain the ranked list by the index (e.g. [8], [7], [3]) and no additional comments\n\n"
        f"Query: {search_query}\n"
        f"Document lists\n"
        f"{document}"
    )
    

def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

    
def get_singleqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention several URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/first_nobel_physics.html<|end_url|>\n\n"
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_multiqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>voice actor of Lara Croft<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results provide names of voice actors for Lara Croft. I need to confirm if Alice David is one of them.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention the company that developed the video game featuring Alice David as Lara Croft.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/lara_croft_voice_actor.html, http://example.com/game_developer.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_gpqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/ppIII_neutrino.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_math_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s math-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/integration_e_x_squared.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_code_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s programming-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do I implement a binary search algorithm in Python?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up the implementation details of binary search in Python.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>binary search algorithm implementation in Python<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/python_binary_search.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_naive_rag_instruction(question, documents="", dataset_name=None, rewritten=False):
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{question}\n"
        "Documents:\n"
        f"{documents}\n"
    )

def get_pathrag_instruction(question, context):
    user_prompt = (
        '---Role---\n\n'
        'You are a helpful assistant responding to questions about data in the tables provided.\n\n'

        '---Goal---\n\n'
        "Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.\n"
        "If you don't know the answer, just say so. Do not make anything up.\n"
        'Do not include information where the supporting evidence for it is not provided.\n\n'

        '---Target response length and format---\n\n'

        f'Multiple Paragraphs\n\n'

        '---Data tables---\n\n'

        f'{context}\n\n'

        'Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.\n'
        'Then conclude with a single-line final answer that directly and concisely answers the question, based on either explicitly stated facts or strong, logical inferences.\n'
        'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'

        f'Question: {question}\n\n'
    )
    return user_prompt

def get_haystack_rag_instruction(question, documents="", dataset_name=None):
    if documents is None or documents == "":
        return f'You will answer a question. Your answer should be short and based on either facts or strong, logical inferences.\n\nQuestion: {question}\n\nReturn only the final answer with no additional explanation or reasoning.'
    else:
        return f'You will answer a question based on the following snippet:\n\n{documents}\n\nUse the information provided in the snippet to answer the question. Your answer should be short and based on either explicitly stated facts or strong, logical inferences.\n\nQuestion: {question}\n\nReturn only the final answer with no additional explanation or reasoning.'


def get_task_instruction_conditional_rewrite_docs(document, question, context):
    user_prompt = (
        'You are given a user query, the current context (a summary of previously seen documents), and a new raw document.\n'
        'You are not answering the query directly.\n'
        'Your task is to maintain and update a context summary that accumulates relevant information across multiple documents, which will later be used to answer the query.\n'
        'For each new raw document, follow the guidelines below to revise the context:\n\n'

        '[Guidelines]\n\n'

        '1. Relevance Check\n'
        '- If the raw document is not relevant to the query, return the current context unchanged.\n'
        '- A document is considered relevant if it directly answers the query or provides logically connected information.\n\n'

        '2. Redundancy & Compression\n'
        '- If the raw document contains information already covered in the current context, keep the context as-is.\n'
        '- If the same idea is expressed differently, prefer the clearer and more concise version.\n\n'

        '3. Conflict Resolution\n'
        '- If the raw document contradicts information in the current context, choose the version that better answers the query.\n'
        '- To resolve conflicts, prioritize based on:\n'
        '    - (a) Direct relevance to the query\n'
        '    - (b) Specificity and clarity\n'
        '    - (c) Trustworthiness or general consensus\n\n'

        '4. Incorporating New Information\n'
        '- If the raw document contains relevant information not yet included in the context, merge it into the context at an appropriate place.\n'
        '- If similar information appears across multiple documents, summarize it in a concise and unified form.\n\n'

        '5. Prioritization\n'
        '- If the context becomes long, prioritize information that more directly and effectively answers the query.\n'
        '- Less essential or tangential information may be omitted or briefly summarized.\n\n'

        '[IMPORTANT RULES]\n\n'

        '- Do **not** attempt to answer the query.\n'
        '- Do **not** mention whether the document is relevant or irrelevant.\n'
        '- Do **not** include statements like "X is not available in the document" or "there is no information about Y."\n'
        '- Do **not** include explanations, reasoning, or any reference to the task.\n'
        '- Do **not** include any sentence that reflects on what is present or missing.\n\n'

        'You must return only the updated context, with no commentary or formatting.\n'

        '[Query]\n'
        f'{question}\n\n'

        '[Current Context]\n'
        f'{context}\n\n'

        '[Raw Document]\n'
        f'{document}\n\n'

        '[Updated Context]\n'
        '(Return the revised context text only.)\n'
    )
    return user_prompt

def get_task_instruction_naive_rewrite_docs(document, question=None, other_documents=None, mode=None, num_of_docs=0,
                                               other_rewritten_docs=False):
    assert question is not None
    if mode == 'conditional_cot2':
        user_prompt = (
            "You are given a user query, a set of other documents, and one target document.\n"
            "Your task is to answer the query based solely on the target document, while using the other documents to help clarify and improve the target document.\n"
            "To do this, you will first rewrite the target document to make it more accurate, complete, and easier to reason with in the context of the query.\n"
            "Step 1: Rewrite the **target document** using the other documents so that it is accurate, non-redundant, and maximally informative in light of the query.\n\n"
            "Step 2: Use the rewritten document to answer the query. Think step by step and explain your reasoning.\n\n"
            
            '[Step 1 Guidelines]\n\n'

            '1. Relevance Check\n'
            '- If parts of the target document are not relevant to the query or not logically connected to it, remove them without explanation.\n'
            '- Partial relevant information is acceptable. The document does not need to be sufficient to fully answer the query.\n'
            '- If the document does not contain any content relevant to the query, return exactly: [NO_REWRITE]\n\n'

            '2. Integration\n'
            '- If information in the target document appears similar to what is in the other documents, do not overwrite or remove information—highlight subtle differences or unique nuances.\n'
            '- If the target document contradicts the other documents, preserve the target document’s version.\n'
            '- When rewriting, preserve original terminology from the target document or reuse vocabulary from the query wherever possible, rather than paraphrasing unnecessarily.\n'
            '- Enrich or clarify the target version using the other documents if helpful, but maintain its perspective as primary.\n\n'

            '3. Focus & Prioritization\n'
            '- Emphasize information that most directly supports answering the query.\n'
            '- Less essential or tangential content may be shortened, but not removed if it offers a unique nuance or framing.\n\n'

            '[IMPORTANT RULES]\n\n'

            '- Do **not** answer the query directly in Step 1.\n'
            '- Do **not** mention the existence of other documents.\n'
            '- Do **not** explain your changes or reference the task.\n'
            '- Do **not** include statements like "target document does not provide" or "according to other documents."\n'
            '- Return only the rewritten version of the target document with no commentary or formatting.\n\n'

            '[Query]\n'
            f'{question}\n\n'

            '[Other Documents]\n'
            f'{other_documents}\n\n'

            '[Target Document]\n'
            f'{document}\n\n'

            "You must follow the output format exactly as specified below.\n"
            "Output format:\n"
            "Step 1. Document Rewriting: <your rewritten version of the target document>\n"
            "Step 2. Answer: <your answer to the query, along with explanation based only on the rewritten document>\n"
        )

    elif mode == 'list_cot2':
        output_format = "\n".join([f'Rewritten Target Document {i+1}: (Rewritten version of Target Document {i+1})' for i in range(num_of_docs)])
        user_prompt = (
            "You are given a user query, a set of target documents.\n"
            "Your task is to answer the query based solely on the target documents.\n"
            "To do this, you will first rewrite the target documents to make it more accurate, complete, and easier to reason with in the context of the query.\n"
            "Step 1: Rewrite the **target documents** so that it is accurate, non-redundant, and maximally informative in light of the query.\n\n"
            "Step 2: Use the rewritten documents to answer the query. Think step by step and explain your reasoning.\n\n"
            
            '[Step 1 Guidelines]\n\n'

            '1. Relevance Check\n'
            '- If parts of the target documents are not relevant to the query or not logically connected to it, remove them without explanation.\n'
            '- Partial relevant information is acceptable. The document does not need to be sufficient to fully answer the query.\n'
            '- If the document does not contain any content relevant to the query, return exactly: [NO_REWRITE]\n\n'

            '2. Integration\n'
            '- If information in the target document appears similar to what is in the other documents, do not overwrite or remove information—highlight subtle differences or unique nuances.\n'
            '- If the target document contradicts the other documents, preserve the target document’s version.\n'
            '- When rewriting, preserve original terminology from the target document or reuse vocabulary from the query wherever possible, rather than paraphrasing unnecessarily.\n'
            '- Enrich or clarify the target version using the other documents if helpful, but maintain its perspective as primary.\n\n'

            '3. Focus & Prioritization\n'
            '- Emphasize information that most directly supports answering the query.\n'
            '- Less essential or tangential content may be shortened, but not removed if it offers a unique nuance or framing.\n\n'

            '[IMPORTANT RULES]\n\n'

            '- Do **not** answer the query directly in Step 1.\n'
            '- Do **not** mention the existence of other documents.\n'
            '- Do **not** explain your changes or reference the task.\n'
            '- Do **not** include statements like "target document does not provide" or "according to other documents."\n'
            '- Return only the rewritten version of the target document with no commentary or formatting.\n\n'

            '[Query]\n'
            f'{question}\n\n'

            '[Target Document]\n'
            f'{document}\n\n'

            "You must follow the output format exactly as specified below.\n"
            "Output format:\n"
            "Step 1. Document Rewriting:\n"
            f"{output_format}"
            "Step 2. Answer: <your answer to the query, along with explanation based only on the rewritten document>\n"
        )

    elif mode == 'trained_rewriter':
        user_prompt = (
            'You are a helpful assistant. Your job is to analyze the documents below and rewrite only the parts that help clarify or refine the information in relation to the question.\n'
            'List each relevant document to better support answering the question. Do not include unrelated documents.\n\n'
            
            'Question:\n'
            f'{question}\n\n'
            
            'Documents:\n'
            f'{document}\n\n'
        )

    else:
        assert False
    return user_prompt



def get_task_instruction_openqa(question, model_name=None, dataset_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_final_output_openqa(question, model_name=None, dataset_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )

    return user_prompt


def get_task_instruction_math(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following math question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following math question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_multi_choice(question, documents="",
                                      model_name=None,
                                      dataset_name=None,
                                      ):
    # if model_name == 'qwq':
    #     user_prompt = (
    #         'Please answer the following multiple-choice question. '
    #         'You should provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
    #         f'Question:\n{question}\n\n'
    #     )
    # elif model_name == 'llama':
    #     user_prompt = (
    #         'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
    #         'Provide your final choice in the format \\boxed{YOUR_CHOICE}. Your final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
    #         f'Question:\n{question}\n\n'
    #     )
    # else:
    #     user_prompt = (
    #         'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
    #         'Provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
    #         f'Question:\n{question}\n\n'
    #     )
    if documents:
        user_prompt = (
                "Please answer the following multiple-choice question using the provided documents.\n\n"
                'Only provide your final choice in the format \\boxed{YOUR_CHOICE}. Your final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
                'Return only the final answer with no additional explanation or reasoning.'
                f'Question:\n{question}\n\n'
                "Documents:\n"
                f"{documents}\n"
            )
    else:
        user_prompt = (
                'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
                'Provide your final choice in the format \\boxed{YOUR_CHOICE}. Your final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
                'Return only the final answer with no additional explanation or reasoning.'
                f'Question:\n{question}\n'
            )
    return user_prompt

def get_task_instruction_code(question, question_title=None, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Generate a correct Python program that passes all tests for the given problem. '
            'You should provide your final code within a Python code block using triple backticks (```python\n'
            'YOUR_CODE\n'
            '```).\n\n'
            f'Problem Title: {question_title}\n\n'
            f'Problem Statement:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. '
            f'You should think step by step to solve it.\n\nQuestion:\n{question}\n\n'
            'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n'
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt