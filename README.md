# TCR-BERT

TCR-BERT is a large language model trained on T-cell receptor sequences, built using a lightly modified BERT architecture with tweaked pre-training objectives. We make significant use of the HuggingFace transformers library (https://huggingface.co/transformers/index.html) in building our model.

## Model availability

### TCR-BERT (pretrained)

TCR-BERT is available through the HuggingFace model hub (https://huggingface.co/models). We make several versions of our model available (see manuscript for full details regarding each model's training and usage):

* Pre-trained on masked amino acid & antigen binding classification: https://huggingface.co/wukevin/tcr-bert
    * This is the model to use if you are interested in analzying TRB sequences
* Pre-trained on masked amino acid: https://huggingface.co/wukevin/tcr-bert-mlm-only
    * This is the model to use if you are interested in analying both TRA and TRB sequences

These models can be downloaded automatically by using Python code like the following:

```python
# Different model classes needed for different models
from transformers import BertModel
# This model was pretrained on MAA and TRB classification
tcrbert_model = BertModel.from_pretrained("wukevin/tcr-bert", use_auth_token=True)
```

This will download the model (or use a cached version if previously downloaded) and load the pre-trained weights as appropriate (see the HuggingFace documentation for more details on this API). We leverage this API within the scripts described in the "Usage" section below, as well as in example Jupyter notebooks.

### Fine-tuned TCR-BERT

We fine-tune our TCR-BERT model to predict LCMV GP33 antigen binding given TRA/TRB amino acid pairs (see manuscript for additional details). This uses an architecture augmented from those natively supported in the `transformers` library; thus, this model cannot be loaded using the above Python API. Instead, download latest version of the model as listed:

| Model version | Description | Link | tar.gz `md5sum` |
| --- | --- | --- | --- |
| 1.0 | Initial release | [Link](https://drive.google.com/file/d/1VZ1qyNmeYu7mTdDmSH1i00lKIBY26Qoo/view?usp=sharing) | `e51d8ae58974c2e02d37fd4b51d448ee`

After downloading and unpacking the model to `some_dir`, the model itself can be loaded using a helper function implemented under `model_utils.py`.

```python
import model_utils
# Returns a skorch.NeuralNet object
net = model_utils.load_two_part_bert_classifier("some_dir")
```

You can then make predictions on new examples using the following code snippet:

```python
import data_loaders as dl

# These two lists should zip together to form your TRA/TRB pairs
tra_sequences = ["CAALYGNEKITF"]  # We show only one sequence for simplicity
trb_sequences = ["CASSDAGGRNTLYF"]

# Create a dataset wrapping these sequences
my_dset = dl.TcrFineTuneDataset(tra_sequences, trb_sequences, skorch_mode=True)
preds = net.predict_proba(my_dset)[:, 1]  # Returns a n_seq x 2 array of predictions, second col is positive
```

Please see our example Jupyter notebooks for additional examples using this model.

## Usage

### Using TCR-BERT to identify antigen specificity groups

TCR-BERT can be used to embed sequences. These embeddings can then be used to cluster sequences into shared specificity groups using the Leiden clustering algorithm. The script `bin/embed_and_cluster.py` performs this with the following positional arguments (use `-h` for a complete list of all arguments):

* Input file: File containing list of sequences. This could be formatted as one sequence per line, or as a tab-delimited file with the sequence as the first column.
* Output file: File containing, on each line, a comma-separated list of sequences. Each line corresponds to one group of sequences predicted to share antigen specificity.

An example of its usage is below. 

```bash
python bin/embed_and_cluster.py example_files/glanville_np177_training_patient.tsv temp.csv -r 128 -g 0
```

In the above, we use the `-r` parameter to adjust the resolution of clustering, which affects the granularity of the clusters. Larger values corresponds to more granular clusters, whereas smaller values create coarser (more inclusive) clusters. The `-g` parameter is used to specify the index of the GPU to run on.

### Training a model to predict TRB antigen binding

The most straightforward, data-efficient way to use TCR-BERT to perform TRB-antigen binding prediction is to use TCR-BERT to embed the TRB sequences, apply PCA to reduce the dimensionality of the embedding layer, and use a SVM to perform classification. This process is automated in the script (under `bin`) `embed_and_train_pcasvm.py`. This script takes three positional arguments:

* Input file: tab-delimited file, where the first column lists TRB sequences, and the second column lists corresponding labels. Other columns are ignored. These inputs will be randomly split into training and test splits unless the `--test` argument is also provided.
* Output directory: Folder to write the resulting PCA-SVM model to.

An example of its usage is below, along with expected output:

```bash
> python bin/embed_and_train_classifier.py example_files/glanville_np177_training_patient.tsv outdir --test example_files/glanville_np177_testing_patients.tsv
INFO:root:Git commit: 4652c66724ea336b95b61a0cb6401013e29218ed
INFO:root:Reading in test set examples
INFO:root:Training: 1055
INFO:root:Testing:  227
WARNING:root:Defaulting to CPU
Some weights of the model checkpoint at wukevin/tcr-bert were not used when initializing BertModel: ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
WARNING:root:Defaulting to CPU
Some weights of the model checkpoint at wukevin/tcr-bert were not used when initializing BertModel: ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
- This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
INFO:root:Test AUROC: 0.6026
INFO:root:Test AUPRC: 0.3521
INFO:root:Writing PCA-SVM model to outdir/pcasvm.sklearn
```

These example files contain TRB sequences that are known to bind to the NP177 influenza A antigen, courtesy of a dataset published by Glanville et al., as well as a set of randomly selected endogenous human TRBs sampled from TCRdb (as a background negative set). Since we split this data by patient (rather than using default of random splits), we explicitly provide a separate set of test examples using the `--test` argument.

## Example Jupyter notebooks

We provide several Jupyter notebooks that demonstrate more interactive, in-depth analysis using TCR-BERT. These notebooks also serve to replicate some of the results we present in our manuscript. Under the `jupyter` directory, please find:

* `transformers_glanville_classifier_and_clustering.ipynb` containing classification analysis and clustering analysis using the Glanville dataset
* `transformers_finetuned_tcr_engineering_mlm_gen.ipynb` containing code used for generating, analyzing, and visualizing novel TCR sequences binding to murine GP33.

## Selected References

J. Glanville et al., Identifying specificity groups in the t cell receptor repertoire. Nature 547(7661), 94–98 (2017).

J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: Pre-training of deep bidirectional transformers for language understanding proceedings of the 2019 conference of the north american chapter of the association for computational linguistics: Human language technologies, volume 1 (long and short papers). Proceedings of the 2019 Conference of the North 4171–4186 (2019).
