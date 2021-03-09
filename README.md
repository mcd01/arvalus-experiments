# arvalus-experiments
Data / Scripts for the paper "Learning Dependencies in Distributed Cloud Applications to Identify and Localize Anomalies" accepted to the CloudIntelligence Workshop at ICSE 2021. We prototypically implemented our method and tested it in a setup with synthetic data. Please consider reaching out if you have questions or encounter problems.

## Technical Details

#### Important Packages
* Python `3.6.0+`
* [PyTorch](https://pytorch.org/) `1.7.0`
* [PyTorch Ignite](https://pytorch.org/ignite/) `0.4.2`
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/1.6.3/) `1.6.3`

All necessary packages will be installed in an environment.

#### Install Environment
install dependencies with: 

    conda env [create / update] -f code/environment.yml

#### Activate Environment
For activation of the environment, execute:

    [source / conda] activate arvalus

## Notes

#### Data Extraction

Extract the *raw.tar.gz* in the data folder, such that you have a folder *raw* in the data folder with all files inside.

#### Preparation for Evaluation
Setup a "results" folder in the root directory of this project. 
Add "plots" subdirectory, and again "classification" and "localization" sub-directories. This is only mandatory if you want to use the evaluation scripts.

## Example-Command

We can execute an experiment (similar to the one described in the paper) with the following command:

    python run.py --data-root ../../data --device cpu --epochs 100 --input-dimension 20 --batch-size 32 --model ModelGCN --seed 73 --num-workers 4 --use-validation --exclude-anomalies bandwidth packet_duplication packet_loss stress_mem mem_leak stress_cpu normal

Some words regarding certain parameters (more details in *run.py*):

* **data-root**: The data directory. Requires to have a sub-directory *raw* that contains the files.
* **device**: The device to use, e.g. *cpu* or *cuda:0*.
* **input-dimension**: The number of features per graph node. Also the window width during extraction of data slices for graph construction (e.g. 20 = 10 seconds window at 2Hz sampling frequency). 
* **model**: The model to choose. Either ModelCNN (Arvalus) or ModelGCN (D-Arvalus).
* **seed**: A seed for reproducible data splitting.
* **num-workers**: The number of worker threads to use for data loading.
* **use-validation**: Indicates that a validation set shall be used.
* **exclude-anomalies**: The anomalies to exclude, if any. Since our dataset originates from an old and different experiment setup, we need to exclude several 'anomalies' (tags in the csv-files) and sequences of global normal behavior in order to arrive at 2 anomalies which we then can synthesize (as described in the paper).
 
If a processed dataset of graphs does not yet exist, this command will first issue the creation of such a dataset (this might take a while). Subsequently, this dataset is used for splitting the data (5 folds in total), training the respective model and evaluating it on the holdout dataset. Eventually, the results are saved to disk.

The evaluation scripts can then be used to inspect the saved results.

