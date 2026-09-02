import logging
import re
import inspect
import tiktoken
import backoff
import openai
import ipdb
import os

from openai.error import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()    
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))        
logger.addHandler(handler)

def set_api(api_name: str, model_name: str=None):
        #self.current_api = api_name
        #self.curr_model_name = model_name

        if api_name == "openai":
            logger.info(f"Using OpenAI API")
            #openai.api_key = os.getenv("OPENAI_API_KEY") # mnskim0
            openai.api_key = os.getenv("OPENAI_API_KEY") # snu.ac.kr
            openai.api_key = os.getenv("OPENAI_API_KEY") # mnskim0 mind2webgpt
            
            os.environ["OPENAI_API_KEY"] = openai.api_key
            openai.api_type = 'open_ai'
            openai.api_base = 'https://api.openai.com/v1'
            openai.api_version = None
        elif api_name == "openai_codex":    
            logger.info(f"Using OpenAI Codex API")
            openai.api_key = os.getenv("OPENAI_API_KEY")
            openai.api_type = 'open_ai'
            openai.api_base = 'https://api.openai.com/v1'
            openai.api_version = None
        elif api_name == "azure1":
            logger.info(f"Using Azure API")
            openai.api_key = os.getenv("AZURE_OPENAI_KEY_1")
            openai.api_type = "azure"
            openai.api_base = "https://ldi2023auginstance1.openai.azure.com/"
            openai.api_version = "2023-05-15"
        elif api_name == "azure2":
            logger.info(f"Using Azure API")
            openai.api_key = os.getenv("AZURE_OPENAI_KEY_2")
            openai.api_type = "azure"
            openai.api_base = ""
            openai.api_version = "2023-05-15"

def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages.
    Borrowed from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        if model == 'gpt-35-turbo-16k-mnskim':
            model = 'gpt-3.5-turbo-16k-0613'
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.info("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo-1106",
        "gpt-35-turbo-mnskim",
        "gpt-35-turbo-16k-mnskim"
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


MAX_TOKENS = {
    "gpt-3.5-turbo-0301": 4097-750,
    "gpt-3.5-turbo-0613": 4097-750,
    "gpt-3.5-turbo-16k-0613": 16385-750,
    "gpt-35-turbo-16k-mnskim": 16385-750,
    "gpt-35-turbo-mnskim": 4097-750,
    "gpt-4-0613": 8192-750,
    "gpt-4-0314": 8192-750,
    "gpt-4-1106-preview": 16385-750,
}


def get_mode(model: str) -> str:
    """Check if the model is a chat model."""

    if model in [
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo-1106",
        "gpt-35-turbo-16k-mnskim",
        "gpt-35-turbo-mnskim"
    ]:
        return "chat"
    elif model in [
        "davinci-002",
        "gpt-3.5-turbo-instruct-0914",
    ]:
        return "completion"
    else:
        raise ValueError(f"Unknown model: {model}")

class MaxRetriesException(Exception):
    """Exception raised when maximum retries are reached."""
    pass

def on_giveup_handler(details):
    error_message = "Max retries reached. Giving up."
    raise MaxRetriesException(error_message)

# NOTE don't catch TimeoutError here, leads to problems
    
@backoff.on_exception(
    backoff.constant,
    (APIError, RateLimitError, APIConnectionError, ServiceUnavailableError, openai.error.Timeout),
    interval=10,
    max_tries=3,
    on_giveup=on_giveup_handler,
)
def generate_response(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    stop_tokens: list[str] | None = None,
) -> tuple[str, dict[str, int]]:
    """Send a request to the OpenAI API."""

    #ipdb.set_trace()

    logger.info(
        f"Send a request to the language model from {inspect.stack()[1].function}"
    )

    #ipdb.set_trace()

    if get_mode(model) == "chat":
        #ipdb.set_trace()
        if openai.api_type == "open_ai":
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stop=stop_tokens if stop_tokens else None,
                request_timeout=15,
            )
        if openai.api_type == "azure":
            
            response = openai.ChatCompletion.create(
                engine=model,
                messages=messages,
                temperature=temperature,
                stop=stop_tokens if stop_tokens else None,
            )
            """
            response = openai.ChatCompletion.create(
                engine=self.curr_model_name,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_token_length,
                top_p=self.top_p,
                stop=self.stop,                                                
            )
            """
        #ipdb.set_trace()    
        if 'content' not in response['choices'][0]['message']:
            #ipdb.set_trace()
            if response['choices'][0]['finish_reason'] == 'content_filter':
                # switch to open_ai
                set_api('openai')
                logger.info(f"Temporarily switching to OpenAI API due to content filter")
                if model == 'gpt-35-turbo-mnskim':
                    _model = 'gpt-3.5-turbo-0613'
                if model == 'gpt-35-turbo-16k-mnskim':
                    _model = 'gpt-3.5-turbo-16k-0613'
                    
                response = openai.ChatCompletion.create(
                    model=_model,
                    messages=messages,
                    temperature=temperature,
                    stop=stop_tokens if stop_tokens else None,
                    request_timeout=15,
                )
                # set back to azure
                set_api('azure1')
            #ipdb.set_trace()

        message = response["choices"][0]["message"]["content"]
    else:
        prompt = "\n\n".join(m["content"] for m in messages) + "\n\n"
        response = openai.Completion.create(
            prompt=prompt,
            engine=model,
            temperature=temperature,
            stop=stop_tokens if stop_tokens else None,
        )
        message = response["choices"][0]["text"]
    info = {
        key: response["usage"][key]
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }

    return message, info


def extract_from_response(response: str, backtick="```") -> str:
    if backtick == "```":
        # Matches anything between ```<optional label>\n and \n```
        pattern = r"```(?:[a-zA-Z]*)\n?(.*?)\n?```"
    elif backtick == "`":
        pattern = r"`(.*?)`"
    else:
        raise ValueError(f"Unknown backtick: {backtick}")
    match = re.search(
        pattern, response, re.DOTALL
    )  # re.DOTALL makes . match also newlines
    if match:
        extracted_string = match.group(1)
    else:
        extracted_string = ""

    return extracted_string
