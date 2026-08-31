
def get_rerank_system_prompt(mode, **kwargs):
    if mode == 'relevance':
        system_prompt = (
            "You are RankLLM, an intelligent assistant that can rank documents based on their relevancy to the query.\n"
        )
    elif mode == 'rearank':
        system_prompt = (
            "You are DeepRerank, an intelligent assistant that can rank passages based on their relevancy to the search query. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        )
    elif mode == 'reasonrank':
        system_prompt = (
            "You are RankLLM, an intelligent assistant that can rank passages based on their relevance to the query. Given a query and a passage list, you first thinks about the reasoning process in the mind and then provides the answer (i.e., the reranked passage list). The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
        )
    elif mode == 'reasoning_and_ranking':
        n_docs = kwargs.get('n_docs', 10)
        system_prompt = (
            "You are an AI assistant that analyzes complex questions and identifies which documents best support answering them.\n\n"

            "Given a user's query and a set of documents, your task is to:\n"
            "1. Generate a reasoning trace, thinking step by step about what knowledge or types of information are necessary to answer the query. These should be abstract but specific enough to guide document selection.\n"
            f"2. Select and rank at least {min(n_docs,10)} documents that best support the reasoning steps. Consider how each document contributes to the reasoning process. Order them from most to least useful using `>` between document IDs (e.g., [3] > [7]).\n\n"

                "Use the following format:\n"
                "[Reasoning Trace]\n"
                "Step 1: <First reasoning step>\n"
                "Step 2: <Second reasoning step>\n"
                "...\n"
                "Step N: <Final reasoning step>\n\n"

                "[Document Ranking]\n"
                "[9] > [5] > [6] > ... > [12]\n\n"

            "Only produce the output in the format shown above."
        )
        # system_prompt = (
        #     "You are an AI assistant that analyzes complex questions and identifies which documents best support answering them.\n\n"

        #     "Given a user's query and a set of documents, your task is to:\n"
        #     "1. Generate a reasoning trace, thinking step by step about what knowledge or types of information are necessary to answer the query. These should be abstract but specific enough to guide document selection.\n"
        #     f"2. Select and rank at least {min(n_docs,10)} documents that best support the reasoning steps. Consider how each document contributes to the reasoning process. Order them from most to least useful using `>` between document IDs (e.g., [3] > [7]).\n\n"

        #     "Use the following format:\n"
        #     "[Reasoning Trace]\n"
        #     "Step 1: <First reasoning step>\n"
        #     "Step 2: <Second reasoning step>\n"
        #     "...\n"
        #     "Step N: <Final reasoning step>\n\n"

        #     "[Document Support Analysis]\n"
        #     "[1]: <short description of how doc 1 supports reasoning trace>\n"
        #     "[2]: <short description of how doc 2 supports reasoning trace>\n"
        #     "...\n"
        #     f"[{n_docs}]: <short description of how doc {n_docs} supports reasoning trace>\n\n"

        #     "[Document Ranking]\n"
        #     "[9] > [5] > [6] > ... > [12]\n\n"

        #     "Only produce the output in the format shown above."
        # )
    else:
        assert False
    return system_prompt

def get_rerank_user_prompt(query, docs, mode, **kwargs):
    if mode == 'relevance':
        n_docs = kwargs.get('n_docs', 20)
        user_prompt = (
            f"I will provide you with {n_docs} documents, each indicated by  a numerical identifier []. Rank the documents based on their relevance to the search query: {query}\n"
            f"{docs}\n\n"

            f"Search Query: {query}\n"
            f"Rank the {n_docs} documents above based on their relevance to the search query. All the documents should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [2] > [1], Only respond with the ranking results, do not say any word or explain.\n"
        )
    elif mode == 'rearank':
        assert isinstance(docs, list)
        messages = []

        instruction =  (
            f"I will provide you with passages, each indicated by number identifier []. Rank the passages based on their relevance to the search query."
            f"Search Query: {query}. \nRank the {len(docs)} passages above based on their relevance to the search query."
            f"The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be <answer> [] > [] </answer>, e.g., <answer> [1] > [2] </answer>."
        )
        messages.append({"role": "user","content": instruction})
        messages.append({"role": "assistant", "content": "Okay, please provide the passages."})

        for idx, doc in enumerate(docs):
            if 'content' in doc:
                contents = doc['content'].strip()
            elif 'contents' in doc:
                contents = doc['contents'].strip()
            else:
                assert False
            contents = ' '.join(contents.split())
            messages.append({"role": "user", "content": f"[{idx+1}] {contents[:400]}"})
            messages.append({"role": "assistant", "content": f"Received passage [{idx+1}]."})

        prompt = (
            f'Please rank these passages according to their relevance to the search query: "{query}"\n'
            "Follow these steps exactly:\n"
            "1. First, within <think> tags, analyze EACH passage individually:\n"
            "- Evaluate how well it addresses the query\n"
            "- Note specific relevant information or keywords\n\n"
            "2. Then, within <answer> tags, provide ONLY the final ranking in descending order of relevance using the format: [X] > [Y] > [Z]"
        )
        messages.append({
            "role": "user",
            "content": prompt
        })
        return messages
    elif mode == 'reasonrank':
        num = kwargs.get('n_docs', 20)
        prefix = f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n"
        user_prompt = (
            f"{prefix}"
            f"{docs}\n\n"
            f"Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The format of the answer should be [] > [], e.g., [2] > [1]."
        )
        return user_prompt

    elif mode == 'reasoning_and_ranking':
        user_prompt = (
            "[Query]\n"
            f"{query}\n\n"

            "[Documents]\n"
            f"{docs}\n\n"
        )
    return user_prompt

