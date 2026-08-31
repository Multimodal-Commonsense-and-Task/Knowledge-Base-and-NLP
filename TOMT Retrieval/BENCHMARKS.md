# Benchmarks 

Note:
- We use [py-terrier](https://github.com/terrier-org/pyterrier) for the bechmarks. This requires a Java environment 
setup (see original repository for setup instructions).
- The lines below only reproduce BM25/PL2. For running DPR, see the README.md in 
the sub-folder `DPR` (this requires a separate environment).

By executing the following script, the test performance of each benchmark is saved in `results/{dataset}/{config}/{method}`:

```
sh reproduce_lexical.sh 
```
