# GP-WAITER
![GP-WAITER](./images/figure1.png)

GP-WAITER (Genome-Phenotype prediction using Weighted self-AttentIon TransformER) is a novel model that integrates GWAS-derived SNP weights into a hybrid CNN-Transformer architecture via an efficient tokenized embedding scheme. The model's design enables dynamic learning of trait-associated feature weights and effective modeling of long-range interactions within ultra-long genomic sequences.
# Project Structure
```
├──train-test.py
├──requirements.txt
├──README.md
├──model
|    ├──.keep
|    ├──GP_WAITER.so
├──Demo
|    ├──Instructions to run on data
|    |    ├──demo_data
|    |    |  ├──demo.phenotype.O.csv
|    |    |  ├──demo.weighted.O.csv
|    |    |  ├──demo.genotype.O.txt
|    |    |____demo.script.py
|    |───Expected output
|            ├──best_results_summary.json
|            ├──parameters
|                ├──best_O.params
|                ├──O_training_log.json
|                ├──best_O_predictions.csv
├──LICENSE
```

# Getting Started
## Environment
### Hardware
You need GPU with at least 8GB cache and CPU.  
### Software
First, please install CUDA==11.3(compatiable with pytorch) and python==1.12.

Then, install nessasary libraries referring to requirements.txt:

```
torch==1.12

tensorboard

numpy

pandas

scikit-learn
```
You can run the command below in the bash:
```
pip install -r requirements.txt
```
 The installation time is short and you needn't wait for a long time.
## Setting Model Hyperparameters
|Hyperparameter|Description|Default Value|
|:-------------|:--------- |:----------|
|num_epochs   |the number of training epochs |200 |
|batch_size   |Batch size for training  |32 |
|learning_rate|Learning rate for the optimizer  |0.001 |
|optimizer    |Optimizer for improving the performance of the model |Adam  |
|N      |the number of encoder stacks|3 |
|num_heads   |the number of attention heads in each encoder stack|12,10,5(three encoder stacks)|
## Model Training-Testing
Please modify the filepaths of genotype, phenotype and SNP weights data in the `train-test.py` file. Running training-testing.py file while importing GP-WAITER model from `./model/GP-WAITER.py`. Then generate trained models and test the models on a test dataset.  
Running instruction:
```
python3 train-test.py
```
# Demo
The demo file contains only a subset of phenotypes, genotype and weighted information. Its purpose is solely to facilitate rapid testing and evaluate the usability of the model. Results obtained from this file do not reflect the model’s optimal performance and are provided for reference only.
You can run the demo.script.py file under Demo path to help understand the training and testing process.

Running instruction:
```
python3 demo.script.py
```
When using the demo script, modify the data paths as needed to ensure that the sample data is correctly loaded.
After Running the demo, output best model parameters, prediction accuracy and training results for all epoches.
The demo was trained on the provided sample file using an RTX3080 GPU, with an estimated training time of approximately 2 minutes.
# Soybean Genotypic Data Preprocessing Workflow

This repository contains the pipeline for processing biallelic SNP datasets as described in our study.

## Requirements
- [PLINK v1.9](https://www.cog-genomics.org/plink/1.9/)
- [Beagle 5.4](https://faculty.washington.edu/browning/beagle/beagle.html)
- Java JRE 8 or higher

## Pipeline Overview

1.  **Quality Control**: Filters SNPs based on:
    * Minor Allele Frequency (MAF) < 0.01
    * Missingness > 20%
    * Biallelic requirement
2.  **Imputation**: Missing genotypes are filled using Beagle 5.4.
3.  **LD Pruning**: Specifically for the `soybean1861` dataset to reduce marker redundancy.
    * Parameters: `50 5 0.4` (Window: 50 SNPs, Step: 5, $r^2$ threshold: 0.4).
# Calculating SNP Weights
Use EMMAX to extract SNP weights based on GWAS p-value.
# Datasets
We provide some datasets to help train and test GP-WAITER. Meanwhile, you can download the datasets below by visiting [the location](https://doi.org/10.5281/zenodo.18476279) .
| Name    |Type     |Filename         |Line(except headers)  |Row(except index)     |
|:-----   |:----    |:--------        |:-----|:---    |
|Mazie244 |Genotype |maize244_gen.csv |244   |308,136 |
|          |Phenotype|maize244_phe.csv|244   |16       |
|Rice529|Genotype    |rice529_gen.csv|529 |659,573     |
|      |Phenotype|rice529_phe.csv|529   |10      |
|Soybean14460|Genotype    |Soybean14460_gen.csv|14460 |39,707     |
|      |Phenotype|Soybean14460_phe.csv|14460   |11      |
|Soybean192|Genotype    |Soybean192_gen.csv|192 |1,355,959     |
|      |Phenotype|Soybean192_phe.csv|192   |16      |
|Wheat406|Genotype    |Wheat406_gen.csv|406 |234,219     |
|      |Phenotype|Wheat406_phe.csv|406   |9      |
# Citation
Please cite:
Li, J., Yu, L., Li, M. et al. Leveraging weighted embedding and Transformer architecture to improve phenotype prediction of complex traits for crops. Nat Commun (2026). https://doi.org/10.1038/s41467-026-71035-5



