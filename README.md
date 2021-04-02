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

# Additional files
## Genome reference
The genome reference files where downloaded from release 95 and release 87 respectively for the genome version 38 and the genome version 37:
```
cd pre-process/
wget ftp://ftp.ensembl.org/pub/release-95/gtf/homo_sapiens/Homo_sapiens.GRCh38.95.gtf.gz
wget ftp://ftp.ensembl.org/pub/release-87/gtf/homo_sapiens/Homo_sapiens.GRCh37.87.gtf.gz
python read_grch.py 37
python read_grch.py 38
```
The final output should be two .csv files named *"grch37.csv"* and *"grch38.csv"* to be found in the same directory as the main code.
## Gene attribute matrix
To obtain the database for the transcription factors download the gene attribute matrix and the gene list from *ENCODE Project Consortium*:
```
cd pre-process/
wget https://maayanlab.cloud/static/hdfs/harmonizome/data/encodetfppi/gene_attribute_matrix.txt.gz
wget https://maayanlab.cloud/static/hdfs/harmonizome/data/encodetfppi/gene_list_terms.txt.gz
python read_gene_attr_matr.py
```
The final output should be a .csv file named *"gene_attribute_matrix.csv"* to be found in the same directory as the main code.
## Gene attribute matrix info biomart
## miRNA gene matrix
The miRNA database was obtained from TargetScan (release 7.2) and processed in the following way:
```
cd pre-process/
wget http://www.targetscan.org/vert_72/vert_72_data_download/Predicted_Targets_Info.default_predictions.txt.zip
unzip Predicted_Targets_Info.default_predictions.txt.zip
python obtain_miRNA_matrix.py
```
## Cancermine
To obtain the database containing the gene roles type the following code:
```
cd pre-process/
wget https://zenodo.org/record/4580271/files/cancermine_collated.tsv?download=1
python read_cancermine.py
```
The final output should be a .csv file named *"cancermine.csv"* to be found in the same directory as the main code.

# Directory structure and files
```
ChimerDriver.py -> main code
ChimerDriver_tools.py -> python module containing all the necessary functions to run the main code

```

# Training

# Testing
