# Attentional_Factorization_Machine

Paper by
Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu and Tat-Seng Chua (2017). [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.comp.nus.edu.sg/~xiangnan/papers/ijcai17-afm.pdf) IJCAI, Melbourne, Australia, August 19-25, 2017.

Authors implemented with Tensorflow, and there are no codes that are implemented with Pytorch.
So I implemented it with Pytorch.

There will be lot of mistakes.

## Environments
* Pytorch
* numpy
 
 ## Dataset
Use the same input format as the LibFM toolkit (http://www.libfm.org/). In this instruction, we use [MovieLens](grouplens.org/datasets/movielens/latest).
The MovieLens data has been used for personalized tag recommendation, which contains 668,953 tag applications of users on movies. We convert each tag application (user ID, movie ID and tag) to a feature vector using one-hot encoding and obtain 90,445 binary features. The following examples are based on this dataset and it will be referred as ***ml-tag*** wherever in the files' name or inside the code.
When the dataset is ready, the current directory should be like this:

* code
    - AFM.py
    - LoadData.py
    - config.py
    - main.py
* data
    - ml-tag
        - ml-tag.train.libfm
        - ml-tag.validation.libfm
        - ml-tag.test.libfm
