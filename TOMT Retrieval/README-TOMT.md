# Reddit-TOMT

This repository contains data / code for ['It's on the tip of my tongue' - A new dataset for 
Known-Item Retrieval](https://doi.org/10.1145/3488560.3498421), published at
 [WSDM2022](https://www.wsdm-conference.org/2022/). 

Authors: [Samarth Bhargav](https://samarthbhargav.github.io/) | [Georgios Sidiropolous](https://twitter.com/gnsidiro/) 
| [Evangelos Kanoulas](https://staff.fnwi.uva.nl/e.kanoulas/) 
- [IRLab, University of Amsterdam](https://irlab.science.uva.nl/)



### Abstract

The tip of the tongue known-item retrieval (TOT-KIR) task involves the 'one-off' retrieval of an item for which a user
cannot recall a precise identifier. The emergence of several online communities where users pose known-item queries to 
other users indicates the inability of existing search systems to answer such queries. Research in this domain is
hampered by the lack of large, open or realistic datasets. Prior datasets relied on either annotation by crowd workers, 
which can be expensive and time-consuming, or generating synthetic queries, which can be unrealistic. Additionally, 
small datasets make the application of modern (neural) retrieval methods unviable, since they require a large number 
of data-points. In this paper, we collect the largest dataset yet with 15K query-item pairs in two domains, namely, 
Movies and Books, from an online community using heuristics, rendering expensive annotation unnecessary while ensuring 
that queries are realistic. We show that our data collection method is accurate by conducting a data study. We further 
demonstrate that methods like BM25 fall short of answering such queries, corroborating prior research. 
The size of the dataset makes neural methods feasible, which we show outperforms lexical baselines, indicating that 
neural/dense retrieval is superior for the TOT-KIR task.

### Bibtex

To cite the paper/dataset:
```
@inproceedings{10.1145/3488560.3498421,
author = {Bhargav, Samarth and Sidiropoulos, Georgios and Kanoulas, Evangelos},
title = { 'It's on the Tip of My Tongue': A New Dataset for Known-Item Retrieval},
year = {2022},
isbn = {9781450391320},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3488560.3498421},
doi = {10.1145/3488560.3498421},
abstract = {The tip of the tongue known-item retrieval (TOT-KIR) task involves the 'one-off' retrieval of an item for 
which a user cannot recall a precise identifier. The emergence of several online communities where users pose known-item
queries to other users indicates the inability of existing search systems to answer such queries. Research in this
domain is hampered by the lack of large, open or realistic datasets. Prior datasets relied on either annotation by crowd
workers, which can be expensive and time-consuming, or generating synthetic queries, which can be unrealistic. 
Additionally, small datasets make the application of modern (neural) retrieval methods unviable, since they require a 
large number of data-points. In this paper, we collect the largest dataset yet with 15K query-item pairs in two domains,
namely, Movies and Books, from an online community using heuristics, rendering expensive annotation unnecessary while 
ensuring that queries are realistic. We show that our data collection method is accurate by conducting a data study. 
We further demonstrate that methods like BM25 fall short of answering such queries, corroborating prior research. 
The size of the dataset makes neural methods feasible, which we show outperforms lexical baselines, indicating that
neural/dense retrieval is superior for the TOT-KIR task.},
booktitle = {Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
pages = {48–56},
numpages = {9},
keywords = {known item retrieval, tip of the tongue known item retrieval},
location = {Virtual Event, AZ, USA},
series = {WSDM '22}
}
```


This repository contains data and code for reproducing the data collection and benchmarks for the Reddit-TOMT dataset. 


## Data

The data can be downloaded from [the data release folder](./data_release/). A description of the data can be found in [DATA.md](DATA.md#)

Only the submission id of the queries are distributed. The text data can be downloaded by
executing the following commands:

```
mkdir -p dataset
cp data_release/Books.zip dataset/
cp data_release/Movies.zip dataset/
cd dataset 
unzip Books.zip
unzip Movies.zip
cd ../
sh fill_data.sh
```  

## Steps to reproduce / extend
### Setup 

This project was run in `python-3.7.10`, it is highly recommended to use this exact version, with a separate environment
using [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv). The DPR benchmark's requirements are separate (see `DPR/README.md`).
The following lines will install all requirements:   

```
pip install -r top_reqs.txt
# install haystack
pip install farm-haystack@git+https://github.com/deepset-ai/haystack.git@da97d813056e5710009ddd5b00fc3531ece9aee7
# download spacy model
python -m spacy download en_core_web_md

```  


### Reproducing the data collection process 

To reproduce the data collection outlined in the paper, please read the [DATA.md](DATA.md) file
for prerequisites, and run the `download_paper_data.sh` script. This collects
a dataset for the years 2017-2020.  

If you want to add to the collection or collect data for other domains or 
develop new methods, etc, see [Collecting Additional Data in DATA.md](DATA.md#Collecting-Additional-Data) 

 
### Benchmarks


See [Benchmarks](BENCHMARKS.md).

#### Open source projects used

We are thankful for the following software and APIs used in this project:

- [Pushshift](https://api.pushshift.io)
- [PRAW](https://praw.readthedocs.io/en/latest/)
- [Wikiplots](https://github.com/markriedl/WikiPlots)
- [IMDbPy](https://github.com/alberanid/imdbpy)
- [LabelStudio](https://labelstud.io/)
- [Terrier](http://terrier.org/) / [pyTerrier](https://github.com/terrier-org/pyterrier)


