# TSNAPred

TSNAPred is a predictor first proposed to identify type-specific nucleic acid(A-DNA, B-DNA, ssDNA, mRNA, tRNA, rRNA) binding residues implemented by LightGBM and Capsule network.

# Requirements

tensorflow == 1.2.0

keras == 2.0.7

lightgbm == 3.2.1

# Dataset

The high-quality dataset of proteins interacting with type-specific nucleic acid was displayed in the data folder. As mentioned in our paper, we divided the dataset into training(70%)-validation(15%)-test(15%) sets, and each protein in our dataset were shown in 3 lines: 

> \>UniProt ID
>
> Protein sequence(.fasta)
>
> Residue annotation (from BioLip)

# Usage

The model folder provides binary classifiers for each binding nucleic acid that has been trained on the training data set. And you can load the model(.pkl file for LightGBM and .h5 file for CapsNet) directly to make predictions.

Before making predictions for proteins, the sequence feature should be generated by `feature_generation_lgb.py` and `feature_generation_capsnet.py`. The output of `feature_generation_lgb.py` would be as input for LightGBM while the output of `feature_generation_capsnet.py` would input to CapsNet.

Then, run the `excution.py` to make the prediction.

What's more, we also show a case study in `./code/case_study.py`. The sequence feature generated by `feature_generation_lgb.py` and `feature_generation_capsnet.py` for the representative protein O69644 were saved at `./feature/case_study`.

The other details can see in the paper and the codes.

# Features

The sequence features used in our paper were generated by the following software:

Relative solvent accessibility(RSA): [ASAquick](http://mamiris.com/software.html)

Secondary Structure(SS): [PSIPRED](http://bioinf.cs.ucl.ac.uk/psipred)

The relative amino acid propensity for binding(RAAP): [Composition Profiler](http://www.cprofiler.org/)

Evolution conservation score(ECO): [HHblits](https://github.com/soedinglab/hh-suite/)

Disorder: [IUPred2A](https://iupred2a.elte.hu/)

Physiochemical properties: [AAindex](https://www.genome.jp/aaindex/)

Position-specific scoring matrix(PSSM): [PSI-BLAST](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.2.26/)







