# GeneDistFinder
Embedding alignment approach to find genes with different alignment patterns across cell development time. 

**NNDistFinder** module computes distributional distances of gene expression for each query cell 
in terms of its own neighbourhood cells vs. reference neighbourhood cells. Cell neighbourhoods are queried using the data structures available in BBKNN package (https://github.com/Teichlab/bbknn). 

Please see Notebook.ipynb for an example analysis using the pan fetal reference and artificial thymic organoid datasets. 

Implemented by: Dinithi Sumanaweera <br>
Acknowledgement: Krzysztof Polanski

Supplementary for research poster "Gene-level alignment of single-cell trajectories" at ISMB 2024 conference. 
