import os

from openai import OpenAI
# from openai.lib.azure import AzureOpenAI
from openai import AzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import openai
import tenacity

class LLM:
    """Wrapper for OpenAI and Azure OpenAI clients.
    """
    def __init__(self, client_type : str):
        """Initializes the LLM client.

        Args:
            client_type (str): The type of client to use. Valid options are: [openai, azure].

        Raises:
            ValueError: If an invalid client type is provided.
        """
        if client_type == 'openai':
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model_name = 'gpt-3.5-turbo'
        elif client_type == 'azure':
            self.client = AzureOpenAI(
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version='2024-02-01',
                azure_deployment='gpt-4o',
                azure_endpoint='https://ldi.openai.azure.com/'
            )
            self.model_name = 'gpt-4o'
        else:
            raise ValueError(f'Invalid client type: [{client_type}]. Valid options are: [openai, azure]')

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(20))
    def __call__(self, prompt : str, raw : bool = False, **kwargs) -> list | ChatCompletion | None:
        """Call the OpenAI API to generate completions for the given prompt.

        Args:
            prompt (str): The prompt to generate completions for.
            raw (bool, optional): If True, returns the raw response from the API. Defaults to False.

        Returns:
            list: A list of completions if raw is False.
            ChatCompletion: The raw response from the API if raw is True.
            None: If the API call fails.
        """
        messages = [
            {
                'role': 'user',
                'content': prompt,
            }
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
        except (openai.BadRequestError, tenacity.RetryError) as e:
            print(e)
            return None
        if raw:
            return response
        ret_list = []
        for choice in response.choices:
            if not choice.message.content:
                continue
            ret = ' '.join(choice.message.content.split())
            ret_list.append(ret)
        return ret_list
