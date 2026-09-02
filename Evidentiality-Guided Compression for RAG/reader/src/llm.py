import os
import time

import openai
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

openai.api_key = os.environ.get('OPENAI_API_KEY')

MODEL = os.environ.get('ECORAG_READER_MODEL', 'gpt-4o-mini')
SLEEP_TIME = 4


@retry(wait=wait_exponential_jitter(initial=5, max=30), stop=stop_after_attempt(6))
def llm(prompt, temperature=0.0, stop=["\n"]):
    """One reader call. Retries are handled by the decorator.

    Note: the experiments were run with greedy decoding. `--temperature` is kept for
    compatibility with the original scripts but the reader is called at temperature 0.
    """
    time.sleep(SLEEP_TIME)
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant. You should answer the question only using the given information. Provide only with the answer, not the explanation."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop,
    )
    return response['choices'][0]['message']['content']
