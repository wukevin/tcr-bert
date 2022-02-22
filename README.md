# TCR-BERT

TCR-BERT is a large language model trained on T-cell receptor sequences, built using a lightly modified BERT architecture with tweaked pre-training objectives. Please see our [pre-print](https://www.biorxiv.org/content/10.1101/2021.11.18.469186v1) for more information.

## Installation

To install TCR-BERT, clone the GitHub repository and create its requisite conda environment as follows (should take <10 minutes).

```bash
conda env create -f environment.yml
```

Afterwards, use `conda activate tcrbert` before running any commands described below.

## Model availability

### TCR-BERT (pretrained)

TCR-BERT is available through the HuggingFace model hub (<https://huggingface.co/models>). We make several versions of our model available (see manuscript for full details regarding each model's training and usage):

* Pre-trained on masked amino acid & antigen binding classification: <https://huggingface.co/wukevin/tcr-bert>
  * This is the model to use if you are interested in analzying TRB sequences
* Pre-trained on masked amino acid: <https://huggingface.co/wukevin/tcr-bert-mlm-only>
  * This is the model to use if you are interested in analying both TRA and TRB sequences

These models can be downloaded automatically by using Python code like the following:

```python
# Different model classes needed for different models
from transformers import BertModel
# This model was pretrained on MAA and TRB classification
tcrbert_model = BertModel.from_pretrained("wukevin/tcr-bert")
```

This will download the model (or use a cached version if previously downloaded) and load the pre-trained weights as appropriate (see the HuggingFace [documentation](https://huggingface.co/transformers/index.html) for more details on this API). To actually use this model to perform a task like predicting which (of 45) antigen peptides a TRB sequence might react to), you can use the following code snippet.

```python
import model_utils  # found under the "tcr" folder

# Load a TextClassificationPipeline using TCR-BERT
tcrbert_trb_cls = model_utils.load_classification_pipeline("wukevin/tcr-bert", device=0)
# For the pipeline, input amino acids are expected to be spaced
df = model_utils.reformat_classification_pipeline_preds(tcrbert_trb_cls([
    "C A S S P V T G G I Y G Y T F",  # Binds to NLVPMVATV CMV antigen
    "C A T S G R A G V E Q F F",      # Binds to GILGFVFTL flu antigen
]))  # Return a dataframe where each column is an antigen, each row corresponds to an input
```

Please see our example Jupyter notebooks for more usage examples with this API.

### Fine-tuned TCR-BERT

We fine-tune our TCR-BERT model to predict LCMV GP33 antigen binding given TRA/TRB amino acid pairs (data from Daniel et al. [preprint](https://www.biorxiv.org/content/10.1101/2021.12.16.472900v1.full)). This uses an architecture augmented from those natively supported in the `transformers` library; thus, this model cannot be loaded using the above Python API. Rather, we make the model available for download at the following link.

| Model version | Description | Link | tar.gz `md5sum` |
| --- | --- | --- | --- |
| 1.0 | Initial release | [Link](https://drive.google.com/file/d/1VZ1qyNmeYu7mTdDmSH1i00lKIBY26Qoo/view?usp=sharing) | `e51d8ae58974c2e02d37fd4b51d448ee`

We also provide a function implemented under `model_utils.py` that provides a single-line call to download (and cache) the latest version of the model, and load the model.

```python
import model_utils
# Returns a skorch.NeuralNet object
net = model_utils.load_two_part_bert_classifier()
# Alternatively, this function can be used to load a manually downloaded
# and unpacked model directory, as below:
# net = model_utils.load_two_part_bert_classifier("model_dir")
```

You can then make predictions on new examples using the following code snippet:

```python
import data_loaders as dl
import model_utils

net = model_utils.load_two_part_bert_classifier()
# These two lists should zip together to form your TRA/TRB pairs
tra_sequences = ["CAALYGNEKITF"]  # We show only one sequence for simplicity
trb_sequences = ["CASSDAGGRNTLYF"]
# Create a dataset wrapping these sequences
my_dset = dl.TcrFineTuneDataset(tra_sequences, trb_sequences, skorch_mode=True)
preds = net.predict_proba(my_dset)[:, 1]  # Returns a n_seq x 2 array of predictions, second col is positive
print(preds)  # [0.26887017]
```

Please see our example Jupyter notebooks for additional examples using this model.

## Usage

TCR-BERT has been tested on a machine running Ubuntu 18.04.3; with the provided conda environment, this software should be compatible with any recent Linux distrubtion. Also note that while TCR-BERT does not have any special hardware requirements, TCR-BERT's runtime benefits greatly from having a GPU available, particularly for larger inputs.

### Using TCR-BERT to identify antigen specificity groups

TCR-BERT can be used to embed sequences. These embeddings can then be used to cluster sequences into shared specificity groups using the Leiden clustering algorithm. The script `bin/embed_and_cluster.py` performs this with the following positional arguments (use `-h` for a complete list of all arguments):

* Input file: File containing list of sequences. This could be formatted as one sequence per line, or as a tab-delimited file with the sequence as the first column.
* Output file: File containing, on each line, a comma-separated list of sequences. Each line corresponds to one group of sequences predicted to share antigen specificity.

An example of its usage is below; this snippet should take under a minute to run (assuming the TCR-BERT model is downloaded already), and its expected output is provided at `example_files/glanville_np177_training_patient_clustered.csv` for reference.

```bash
❯ python bin/embed_and_cluster.py example_files/glanville_np177_training_patient.tsv temp.csv -r 128 -g 0
INFO:root:Read in 1055 unique valid TCRs from example_files/glanville_np177_training_patient.tsv
Some weights of the model checkpoint at wukevin/tcr-bert were not used when initializing BertModel: ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
/home/wukevin/miniconda3/envs/tcrbert/lib/python3.9/site-packages/anndata/_core/anndata.py:119: ImplicitModificationWarning: Transforming to str index.
  warnings.warn("Transforming to str index.", ImplicitModificationWarning)
INFO:root:Writing 622 TCR clusters to: temp.csv
❯ md5sum temp.csv example_files/glanville_np177_training_patient_clustered.csv
dd9689e632dae8ee38fc74ae9f154061  temp.csv
dd9689e632dae8ee38fc74ae9f154061  example_files/glanville_np177_training_patient_clustered.csv
```

In the above, we use the `-r` parameter to adjust the resolution of clustering, which affects the granularity of the clusters. Larger values corresponds to more granular clusters, whereas smaller values create coarser (more inclusive) clusters. The `-g` parameter is used to specify the index of the GPU to run on. This example should finish in less than 5 minutes on a machine with or without GPU availability.

### Training a model to predict TRB antigen binding

#### TCR-BERT as a black-box embeddings generator

The most straightforward, data-efficient way to use TCR-BERT to perform TRB-antigen binding prediction is to use TCR-BERT to embed the TRB sequences, apply PCA to reduce the dimensionality of the embedding layer, and use a SVM to perform classification. This process is automated in the script (under `bin`) `embed_and_train_pcasvm.py`. This script takes three positional arguments:

* Input file: tab-delimited file, where the first column lists TRB sequences, and the second column lists corresponding labels. Other columns are ignored. These inputs will be randomly split into training and test splits unless the `--test` argument is also provided.
* Output directory: Folder to write the resulting PCA-SVM model to.

An example of its usage is below, along with expected output:

```bash
❯ python bin/embed_and_train_classifier.py example_files/glanville_np177_training_patient.tsv temp -t example_files/glanville_np177_testing_patients.tsv -c svm -g 0
INFO:root:Git commit: cfb7dd4451672683c5248b13cb5d98fe470b9f51
INFO:root:Reading in test set examples
INFO:root:Training: 1055
INFO:root:Testing:  227
Some weights of the model checkpoint at wukevin/tcr-bert were not used when initializing BertModel: ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of the model checkpoint at wukevin/tcr-bert were not used when initializing BertModel: ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
INFO:root:Classifier SVC(probability=True, random_state=6489)
INFO:root:Test AUROC: 0.6240
INFO:root:Test AUPRC: 0.4047
INFO:root:Writing svm model to temp/svm.sklearn
```

These example files contain TRB sequences that are known to bind to the NP177 influenza A antigen, courtesy of a dataset published by Glanville et al., as well as a set of randomly selected endogenous human TRBs sampled from TCRdb (as a background negative set). Since we split this data by patient (rather than using default of random splits), we explicitly provide a separate set of test examples using the `--test` argument.

#### Fine-tuning TCR-BERT

In addition to training an SVM on top of TCR-BERT's embeddings, you can also directly fine-tune TCR-BERT (provided you have enough data to do so effectively). To do so, use the script under `bin/fintune_transformer_single.py`. As an input data file, provide a tab-separated `tsv` file with a column named `TRB` (or `TRA`) and a column named `label`. The `label` column should contain the TCR that the corresponding TCR binds to. If a TCR binds to multiple antigens, this can be expressed by joining the multiple antigens using a comma. Doing so will also automatically result in a multi-label classification problem (instead of the default multi-class classification problem). An example of the usage is as below:

```bash
python bin/finetune_transformer_single.py -p wukevin/tcr-bert --data data/pp65.tsv -s TRB -o finetune_pp65
```

In the above, the `-p` option controls the pretrained model being used as a starting point, `--data` denotes the data file being used, `-s` indicates that finetuning for TRBs (controls how input files are read), and `-o` indicates the output folder for the resulting model and logs. The data file indicated above contains a sample input of sequences from the PIRD and VDJdb databases that bind and do not bind the pp65 antigen. (Note: this resulting model and approach isn't used in our manuscript, and is simply provided here for completeness.)

## Example Jupyter notebooks

We provide several Jupyter notebooks that demonstrate more interactive, in-depth analysis using TCR-BERT. These notebooks also serve to replicate some of the results we present in our manuscript. Under the `jupyter` directory, please find:

* `antigen_cv.ipynb` containing code and results for running antigen cross validation analyses.
* `lcmv_test_set_comparison.ipynb` containing classifier performance on the held out LCMV test set.
* `transformers_lcmv_clustering.ipynb` containing clustering performance on the held out LCMV test set (for TCRDist3 reference, see the corresponding notebook under the `external_eval` folder.).
* `transformers_glanville_classifier_and_clustering.ipynb` containing classification analysis and clustering analysis using the Glanville dataset
* `transformers_finetuned_tcr_engineering_mlm_gen.ipynb` containing code used for generating, analyzing, and visualizing novel TCR sequences binding to murine GP33.

To run these, you must first install jupyter notebook support under the `tcrbert` conda environment (these packages are not included by default to save space). You can do this by running `conda activate tcrbert && conda install -c conda-forge notebook`.

## Selected References

J. Glanville et al., Identifying specificity groups in the t cell receptor repertoire. Nature 547(7661), 94–98 (2017).

J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: Pre-training of deep bidirectional transformers for language understanding proceedings of the 2019 conference of the north american chapter of the association for computational linguistics: Human language technologies, volume 1 (long and short papers). Proceedings of the 2019 Conference of the North 4171–4186 (2019).
