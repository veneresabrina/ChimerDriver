# ChimerDriver

# Citation

Please cite the following paper if this code was useful for your research:

*blabla*

Download from here (PDF): 

```
@article{,
    author =       "",
    title =        "{}",
    booktitle =    "{},
    pages =        "",
    year =         "",
}

```
# Prerequisites
The code is tested under Python 3.6.12 with TensorFlow (GPU) 2.2.0 and Keras 2.4.3. backend, Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
The environment can be created with conda by entering the following commands:
```
conda create --name ChimerDriver python=3.6.12 
conda activate ChimerDriver
conda install -c conda-forge pandas=1.1.5
conda install -c conda-forge scikit-learn=0.23.2
conda install -c conda-forge keras=2.4.3
conda install -c conda-forge matplotlib=3.3.2
```
The above requirements are also listed in the requirements.txt files.

# Directory structure and files
```
ChimerDriver.py -> main code
ChimerDriver_tools.py -> python module containing all the necessary functions to run the main code
processed_db.zip:
    gene_attribute_matrix.csv -> database containing the transcription factors for each gene
    gene_attribute_matrix_info_biomart.txt -> database containing the gene ontologies for each gene
    miRNA_gene_matrix.csv -> database containing the microRNA probabilities for each gene
    cancermine.csv -> database for the roles of genes, either driver, tumor suppressor, oncogenic or other
```
# Build the features
The features for the training set and the test set must be constructed using the following four arguments: 
- "build"
- "file.csv"
- "train/train_again/test" - train if the set will serve as a training set, train_again if the dataset will be merged with a previously built train set and will serve as a trainig set, test otherwise
- "1/0/N" - 1 if the label of the entire dataset is positive (e.g oncogenic gene fusion), 0 if it is negative, N if the "Label" column is already provided in the dataset
Note: the training set must be built before any of the testing sets
```
Build the training set features -> python ChimerDriver.py build DEEPrior_data/training_set.csv train N
Build the validation set features -> python ChimerDriver.py build DEEPrior_data/test_set_1.csv test N
Build the test set features -> python ChimerDriver.py build DEEPrior_data/test_set_2_con_non_onco.csv test N
```
# Training
To cross validate the model with 10-fold cross validation on the provided training set the command line takes the following arguments:
- "train"
- "trainset.csv"
- "testset.csv"
- "valset.csv"
- "num_epochs" - max number of training epochs
- "forest/subset/subset_forest" - either use the random forest selection to reduce the number of features, use a subset of features or combine the two to obtain a feature set made of the selected databases and reduced by the random forest method
- "5/TF/GO/miRNA/5_TF/5_miR/5_GO/TF_miR/GO_TF/GO_miR/5_TF_miR/5_GO_miR/5_GO_TF/5_GO_TF_miR" - pick the features from the transcription factor database (TF), the gene ontologies database (GO), the microRNA database (miRNA), the structural features set (5) or any provided combination of these sets.    
- "forest_thresh" - threshold for the random forest feature selection
- "lr" - learning rate
- "dropout" - dropout value

Train the model on training.set.csv, validate it using the samples in test_set_1.csv. The maximum possible training epochs is 3000, the model uses the random forest selection method on the complete set of features with a threshold of 0.0005. The learning rate is 0.01 and the dropout is 0.2
```
python ChimerDriver.py train DEEPrior_data/training_set.csv DEEPrior_data/test_set_2_con_non_onco.csv DEEPrior_data/test_set_1.csv 3000 forest all 0.0005 0.01 0.2
```
Train the model on training.set.csv, validate it using the samples in test_set_1.csv. The maximum possible training epochs is 1000, the model uses the transcription factor features only. The learning rate is 0.001 and the dropout is 0.1
```
python ChimerDriver.py train DEEPrior_data/training_set.csv DEEPrior_data/test_set_2_con_non_onco.csv DEEPrior_data/test_set_1.csv 1000 subset TF 0 0.01 0.1
```
Train the model on training.set.csv, validate it using the samples in test_set_1.csv. The maximum possible training epochs is 1000, the model uses all the features except for the miRNAs and reduces the number of features with a random forest characterized by a threshold equal to 0.0005. The learning rate is 0.001 and the dropout is 0
```
python ChimerDriver.py train DEEPrior_data/training_set.csv DEEPrior_data/test_set_2_con_non_onco.csv DEEPrior_data/test_set_1.csv 1000 subset 5_GO_TF 0.005 0.01 0
```


# Testing
The command line arguments are the same used for the training phase with the exception of the first one which will be "test" instead of "train"
