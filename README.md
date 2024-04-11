# GeneDistFinder
Alignment free approach to find genes with different alignment patterns across cell development time. 

**NNDistFinder** module computes distributional distances of gene expression for each query cell 
in terms of its own neighbourhood cells vs. reference neighbourhood cells. Cell neighbourhoods are queried using the data structures available in BBKNN package (https://github.com/Teichlab/bbknn). 

Please see notebook.ipynb for an example analysis using the pan fetal reference and artificial thymic organoid datasets. 

Implemented by: Dinithi Sumanaweera <br>
Acknowledgement: Krzysztof Polanski
