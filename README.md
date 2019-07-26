# rwe

Repository containing data (pre-trained embeddings) and code of the ACL-19 paper *[Relational Word Embeddings](https://arxiv.org/abs/1906.01373)*  (ACL 2019). With the code of this repository you can learn your own relational word embeddings from a text corpus.

### Pre-trained embeddings

We release the 300-dimensional embeddings trained on the English Wikipedia used in our experiments:
- [**FastText**](https://drive.google.com/file/d/1SVB7E41c-xvwy61YL3hoDJHRi3RCgf-E/view?usp=sharing) word embeddings \[~300MB\].
- [**Relative-init**](https://drive.google.com/file/d/17bxqdjmn6ZHWgwlstO5d1--3kVf4uQ0N/view?usp=sharing) relation embeddings (symmetrical): \[~6.0GB\]
- Output [**RWE**](https://drive.google.com/file/d/1UjjEb6-80bbJ3GFMFhkRkjvWGULgKpfe/view?usp=sharing) relational word embeddings *(as in the reference paper)*: \[~300MB\]
- Output [**RWE**]() relational word embeddings *(with default parameters using the code below)*: \[~300MB\]

*Note 1:* All vocabulary words are lowercased.

*Note 2:* If you want to convert the *txt* files to *bin*, you can use [convertvec](https://github.com/marekrei/convertvec).

*Note 3:* Underscore "_" is used to separate tokens in a multiword expression (e.g. united_states) in the corpus. Double underscore ("__") is used to separate words within the word pair (e.g. paris__france) in the relation embedding files.

### Code

This repository contains the code to learn unsupervised relation word embeddings in a given text corpus. 

**Requirements:**

- Python (version 3.6.7 tested)
- NumPy (version 1.16.4 tested)
- PyTorch (version 1.1.0 tested)

### Quick start: Get your own relational word embeddings

```bash
python -i train_RWE.py -word_embeddings INPUT_WORD_EMBEDDINGS -rel_embeddings INPUT_RELATION_EMBEDDINGS -output OUTPUT_RWE_EMBEDDINGS
```

The code takes as input standard word embeddings ([FastText](https://github.com/facebookresearch/fastText) with standard hyperparameters was used in the reference paper) and relation embeddings (i.e. embeddings for pairs of words), both in standard space-sparated *txt* formats (see pre-trained embeddings for exact format). As input relation embeddings we used the [Relative package](https://github.com/pedrada88/relative), mainly due to its efficiency compared to other similar methods, but any relation embeddings can be used as input. To learn your own Relative relation embeddings you can simply run the following command (more information in the original [Relative repository](https://github.com/pedrada88/relative)):

```bash
python relative_init.py -corpus INPUT_CORPUS -embeddings INPUT_WORD_EMBEDDINGS -output OUTPUT_RELATIVE_EMBEDDINGS -symmetry true
```
where INPUT_CORPUS can be any tokenized corpus (English Wikipedia in our experiments).

#### Example usage:

A short example on how to use the RWE code:

```bash
python -i train_RWE.py -word_embeddings fasttext_wikipedia_en_300d.txt -rel_embeddings relative-init_symm_wiki_en_300d.txt -output rwe_embeddings.txt
```

### Parameters

A number of optional hyperparameters can be specified in the code. Below you can find these parameters and their default values:

-hidden: Size of the hidden layer. Default: 0 (=twice the dimension of the input word embeddings)

-dropout: Dropout rate. Default: 0.5

-epochs: Number of epochs. Default: 5

-interval: Size of intervals during training. Default: 100

-batchsize: Batch size. Default: 10

-devsize: Size of development data (proportion with respect to the full training set, from 0 to 1). Default: 0.015

-lr: Learning rate for training. Default: 0.01

#### Example:

For example, if you would like more epochs (e.g. 10) and a higher learning rate (e.g. 0.1), you can type the following:

```bash
python -i train_RWE.py -word_embeddings fasttext_wiki_300d.txt -rel_embeddings relative-init_symm_wiki_en_300d.txt -output rwe_embeddings.txt -epochs 10 -lr 0.1
```

*Note:* This code has been tested on GPU for a higher speed, but could be run on CPU as well.

### Reference paper

If you use any of these resources, please cite the following [paper](https://arxiv.org/pdf/1906.01373.pdf):
```bash
@InProceedings{camachocollados:rweacl2019,
  author = 	"Camacho-Collados, Jose and Espinosa-Anke, Luis and Schockaert, Steven",
  title = 	"Relational Word Embeddings",
  booktitle = 	"Proceedings of ACL",
  year = 	"2019",
  location = 	"Florence, Italy"
}

```
If you use [FastText](https://github.com/facebookresearch/fastText) or [Relative](https://github.com/pedrada88/relative), please also cite their corresponding paper/s as well.

License
-------

Code and data in this repository are released open-source.

Copyright (C) 2019, Jose Camacho Collados.
