from langchain_core.prompts import PromptTemplate
# from langchain.llms import HuggingFacePipeline
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch
import argparse
import json
from tqdm import tqdm
from langchain.output_parsers import RetryOutputParser
import datetime
# os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6,7"


def is_valid_format(output, question, prompt_type = "batch", num_groups = 1):
    """
    Check the format of the related questions output.

    Args:
    output (str): The output string containing related questions.

    Returns:
    bool: True if the format is correct, False otherwise.
    """
    # Split the output into lines
    lines = output.split(f"\nQuestion: {question}")[-1].strip()
    lines = lines.split("\n")
    # Check if there are any related questions
    if not lines:
        return False

    while lines:
        if lines[0].startswith("1. 3"):
            break
        lines = lines[1:]
    for i, line in enumerate(lines, 1):
        # Check if each line starts with a number followed by a dot and a space
        if not line.startswith(f"{i}. "):
            if i > 3:
                return True
            else:
                return False

    return True

class RetryGeneration():
    def __init__(self, chain, max_retries=5):
        self.chain = chain
        self.max_retries = max_retries
    
    def __call__(self, context, question, prompt_type, num_groups = 1):
        for i in range(self.max_retries):
            output = self.chain.invoke({"context": context, "question": question})
            if is_valid_format(output, question, prompt_type, num_groups):
                return output
        print('Failed to generate valid output after {} retries'.format(self.max_retries))
        print('Last output:', output)
        return output


parser = argparse.ArgumentParser(description="Time-Sensitive QA with Llama")
parser.add_argument("--icl", type=int, default=0, help="Use ICL template. Default is 0. number of examples")
parser.add_argument("--prompt-type", type=str, default="sp", choices=["sp", "icl", "stepback", "batch"], help="Prompt type")
parser.add_argument("--model-path", type=str, default="../../pretrained_models/Meta-Llama-3-70B-Instruct", help="Model name")
parser.add_argument("--api_call", action="store_true", help="Use API call")
parser.add_argument("--data-path", type=str, default="dataset/test.hard.json", help="Data path")
parser.add_argument("--output-path", type=str, default="output/tl.json", help="Output path")
parser.add_argument("--num-samples", type=int, default=0, help="Number of samples")
parser.add_argument("--do-refine", action="store_true", help="Do refine")
args = parser.parse_args()


max_new_tokens = 1024
# model_path = "../../pretrained_models/Meta-Llama-3-70B-Instruct"
if args.api_call:
    hf = Ollama(# model="llama3",
        model="llama3:70b-instruct-fp16",
        base_url="http://147.46.215.218:3000/ollama",
    headers={
        "Authorization": "Bearer sk-53bbeaf2ec104bb5a66b03c7762b4c57",
        "Content-Type": "application/json"
    },
    ) 
else:
    load_in_8bit = True if '70b' in args.model_path else False
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", load_in_8bit=load_in_8bit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    pipe = pipeline(
        "text-generation", model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        model_kwargs={"torch_dtype": torch.bfloat16} if not load_in_8bit else None,
    )
    hf = HuggingFacePipeline(pipeline=pipe)

if "easy" in args.data_path:
    args.output_path = os.path.join(args.output_path.split("/")[0], "TL", "easy", args.output_path.split("/")[-1])
if not args.do_refine:
    # [basename]/[prompt_type]/[model_name]/[datetime yy mm dd hh]-[num_samples].json
    args.output_path = os.path.join(args.output_path.split("/")[0], "TL", args.prompt_type, args.model_path.rsplit('/', 1)[1], f"{datetime.datetime.now().strftime('%y%m%d%h')}-{args.num_samples}.json")

    args.output_path = args.output_path.replace(".json", f"_prompt_{args.prompt_type}_{args.model_path.rsplit('/', 1)[1]}.json")
    if args.num_samples > 0:
        args.output_path = args.output_path.replace(".json", f"_samples_{args.num_samples}.json")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

# input = {"idx": "/wiki/Ian_Gibson_(politician)#P39#0", "question": "What was the position of Ian Gibson (politician) from May 1997 to May 2001?", "context": "Ian Gibson ( politician ) Ian Gibson ( 26 September 1938 \u2013 9 April 2021 ) was a British scientist and Labour politician who served as Member of Parliament ( MP ) for Norwich North from 1997 to 2009 . Gibson was a professor at the University of East Anglia before his election to Parliament and was viewed as one of the countrys leading cancer experts for his work . He chaired the Science and Technology Committee in the House of Commons from 2001 to 2005 and came into conflict with the Blair government as a member of the Labour left . He resigned from the Commons in 2009 after being barred from standing as a Labour candidate in the 2010 general election , due to his behaviour during the parliamentary expenses scandal . From 2009 to 2013 , he was a journalist at the Norwich Evening News . Early life and academic career . Ian Gibson was born in Dumfries , Scotland , the son of William and Winifred Gibson , in September 1938 . He was educated at the Dumfries Academy before attending the University of Edinburgh , where he acquired a Bachelor of Science degree in Genetics and a doctorate . Whilst at university in Scotland he also turned out for professional Scottish sides Airdrie , St Mirren and Queen of the South . He continued his studies in the US at Indiana University Bloomington and the University of Washington . During the 1960s , he was a keen amateur footballer and played for local side Wymondham Town F.C . as a left-back and captain of the team . Gibson worked for the University of East Anglia from 1965 until his election to Parliament and was made an honorary professor in 2003 . He worked as a research scientist until 1971 , when he became a senior biology lecturer and was promoted to Dean of the School of Biological Sciences in 1991 . Whilst a Dean , he headed a research team investigating cancer because of his expertise in the biology of cancer and drugs targeting leukaemia . Upon his suggestion , Norwich City goalkeeper Bryan Gunn funded the Francesca Gunn laboratory , named after the daughter he lost to leukaemia , which he ran for five years . He was awarded a Champion Award by Macmillan Cancer Support for his work . He was not known to have any political views before his employment at the university but became active in the union ASTMS and fought for better pay for technical and other less senior staff . Parliamentary career . Gibson was a member of the Manufacturing , Science and Finance union executive from 1972 to 1996 , and joined the Labour Party in 1983 after seven years in the Socialist Workers Party . He unsuccessfully contested the marginal Norwich North constituency at the 1992 general election , losing to the Conservative incumbent Patrick Thompson by 266 votes . However , he won the seat at the 1997 general election , following Thompsons retirement , with a majority of over 9,000 . Gibson was successfully re-elected in 2001 and 2005 , albeit with a reduced majority of between 5,000 and 6,000 at both elections . He made his maiden speech on 17 June 1997 . Joining the Science and Technology Select Committee upon his election , Gibson used his platform to push the case for science and better cancer treatment . He chaired the committee from 2001 to 2005 , despite opposition from Labour whips , as well as the All-Party Parliamentary Group on Cancer . Gibson raised many medical and biological issues in the House of Commons , including Gulf War syndrome , which he urged the government to recognise . He later became a member of other committees , including the Innovation , Universities , Science and Skills Select Committee and the East of England Regional Select Committee . Although well-liked on all sides , Gibson was often in conflict with the Blair government , and a problem for the government whips due to his left-wing views . He was a major campaigner against top-up fees for universities and voted against parts of the governments counterterrorism legislation , leading him to clash with fellow Norwich MP Charles Clarke . He was also the coach of the cross-party parliamentary football team . From 1999 to 2005 , he was the teams joint manager . He suffered a minor stroke in September 2004 on a visit to Ramallah in the West Bank , leading him to call for stroke support to be prioritised by health services in the same way as cancer care . In 2006 , he took part in a BBC television Inside Out programme , Clouds of Secrecy . The programme reported on the high incidence of oesophageal cancer in the Norfolk area and the possible link with secret experiments the Ministry of Defence carried out in the 1960s with poisonous chemicals . Gibson expressed the hope that further research might take place to establish whether there was indeed a link between these tests and the high local rates of cancer of the oesophagus . In the same year , he attracted controversy after claiming inbreeding in his constituency may have played a part in its rising number of diabetes cases , but later apologised for these remarks . Gibson announced his intention to stand in the next election in 2006 , stating that he would rather die with my boots on and go missing in action than crawl into early retirement and wear slippers and pantaloons . Expenses controversy . In May 2009 , Gibson surprised many when he became embroiled in the MPs expenses scandal , reportedly claiming for a flat in which his daughter lived rent-free before selling it to her for half its market value . He was then barred from standing in the 2010 general election by a Labour Party disciplinary panel , which his constituency chairman dubbed as a kangaroo court . Believing his position was untenable following the panels decision , he resigned as an MP through the traditional procedural device of becoming Crown Steward and Bailiff of the three Chiltern Hundreds of Stoke , Desborough and Burnham . In the subsequent 2009 Norwich North by-election , Conservative Chloe Smith won the seat over the Labour candidate , who saw the partys vote share drop by almost 27% . After Parliament . Following his resignation , Gibson continued to campaign on local and environmental issues in Norwich and returned to lecturing at institutions including Harvard University . He became president of the Norwich Palestine Solidarity Campaign in 2015 . That year , he was pleased when Jeremy Corbyn , an old friend , was elected as Labour leader ; however , he was later alleged to have said that Corbyn lacked leadership and style , after becoming disillusioned with his performance as leader . Personal life . Gibsons first wife was Verity , a social worker with whom he had two children . He is survived by his daughter , Dominique , but their other daughter , Ruth , died in 1993 . In 1977 , he married Liz , with whom he had another daughter . He was a supporter of Norwich City F.C . and served as the president of his former club , Wymondham Town . In later life , he also became a member of a Norwich theatre group . According to Whos Who , he listed his recreations as football coaching , watching , listening and questioning . Gibson died of pancreatic cancer on 9 April 2021 at the age of 82 . Publications . - with Des Turner . - with Elaine Sherriffs . External links . - Profile from The Guardian Ask Aristotle - Obituary from BBC News - Obituary from The Times", "targets": ["Member of Parliament ( MP ) for Norwich North"]}
inputs = []
with open(args.data_path, "r") as f:
    inputs = [json.loads(line) for line in f]
if args.num_samples > 0:
    inputs = inputs[:args.num_samples]

def initial_answer(args):

    template += """Transform this context to the temporal logic formula. Write only the temporal logic formula without additional explanation. \n\n\n Context: {context}\n\nTemporal logic formula for the context: """

    prompt = PromptTemplate(template=template, input_variables=["context"])

    chain = prompt | hf
    generator = RetryGeneration(chain, max_retries=5)

    with open(args.output_path, "w") as f:
        for input in tqdm(inputs):
            context = input['context']
            question = input['question']
            # output = chain.invoke({"question": question, "context": context})
            output = generator(context, question, args.prompt_type)
            # get the answer from the output
            output = output.split(f"\nQuestion: {question}")[-1].strip()
            json_output = {"idx": input["idx"], "answer": output}
            f.write(json.dumps(json_output) + "\n")


if __name__ == "__main__":
    initial_answer(args)