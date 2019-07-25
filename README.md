# rwe

Repository containing data and code of the ACL-19 paper *[Relational Word Embeddings](https://arxiv.org/abs/1906.01373)*  (ACL 2019).


### Pre-trained embeddings

We release the 300-dimensional embeddings trained on the English Wikipedia used in our experiments:
- [**FastText**](https://drive.google.com/file/d/1SVB7E41c-xvwy61YL3hoDJHRi3RCgf-E/view?usp=sharing) word embeddings \[300MB\].
- [**Relative-init**](https://drive.google.com/file/d/17bxqdjmn6ZHWgwlstO5d1--3kVf4uQ0N/view?usp=sharing) relation embeddings (symmetrical): \[6.0GB\]
- Output [**RWE**](https://drive.google.com/file/d/1UjjEb6-80bbJ3GFMFhkRkjvWGULgKpfe/view?usp=sharing) relational word embeddings: \[300MB\]

*Note 1:* All vocabulary words are lowercased.

*Note 2:* If you want to convert the *txt* files to *bin*, you can use [convertvec](https://github.com/marekrei/convertvec).

*Note 3:* Underscore "_" is used to separate tokens in a multiword expression (e.g. united_states) in the corpus. Double underscore ("__") is used to separate words within the word pair (e.g. paris__france) in the relation embedding files.

### Code

Coming soon.

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
If you use [FastText](https://github.com/facebookresearch/fastText) or [Relative](https://github.com/pedrada88/relative), please also cite its corresponding paper.

License
-------

Code and data in this repository are released open-source.

Copyright (C) 2019, Jose Camacho Collados.
