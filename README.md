# DEKT

Source code and data set for our paper : Dual-State Personalized Knowledge Tracing with Emotional Incorporation.

The code is the implementation of DEKT model, and the data set is the public data set [**ASSISTments Challenge**]([The 2017 ASSISTments Datamining Competition - Dataset](https://sites.google.com/view/assistmentsdatamining/dataset)).



## Dependencies:

- python >= 3.7
- pytorch >=1.12.1
- tqdm
- utils
- pandas
- sklearn



## 

## Dataprocess

First, download the data file:  [**ASSISTments Challenge**]([The 2017 ASSISTments Datamining Competition - Dataset](https://sites.google.com/view/assistmentsdatamining/dataset)), then put it in the folder 'data/'.

Then, preprocess the emotional attributes: run dataprocess/preone.py, pretwo.py and prethree.py to preprocess the emotional attributes, and run datapre.py to divide the original dataset into train set, validation set and test set.

`/dataprocess/`

​	`--- preone.py`

​	`--- pretwo.py`

​	`--- prethree.py`

​	`--- datapre.py`

To simplify the preprocessing process, we discretize the emotional attributes in the form of hyperparameters.



The ASSISTments dataset provides six available relevant emotions, which researchers can select and use according to their needs.

| Frustration | Confusion | Concentration | Boredom | Gaming | Off‐task |
| ----------- | --------- | ------------- | ------- | ------ | -------- |

A detailed description of the emotional attributes can be found in this article: [Affect.](https://files.eric.ed.gov/fulltext/EJ1127034.pdf)



Modify the hyperparameters in the main file, then run the main.py to `train` and `test` the model with one command.

```pythin
python main.py
```

