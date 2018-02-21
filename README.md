# Graph Convolutional Networks for relational graphs

Keras-based implementation of Relational Graph Convolutional Networks for semi-supervised node classification on (directed) relational graphs.

For reproduction of the *entity classification* results in our paper [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (2017) [1], see instructions below.

The code for the *link prediction* task in [1] can be found in the following repository: https://github.com/MichSchli/RelationPrediction

## Installation

```python setup.py install```

## Dependencies

Important: keras 2.0 or higher is not supported as it breaks the Theano sparse matrix API. Tested with keras 1.2.1 and Theano 0.9.0, other versions might work as well.

  * Theano (0.9.0)
  * keras (1.2.1)
  * pandas
  * rdflib
  
Note: It is possible to use the TensorFlow backend of keras as well. Note that keras 1.2.1 uses the TensorFlow 0.11 API. Using TensorFlow as a backend will limit the maximum allowed size of a sparse matrix and therefore some of the experiments might throw an error.

## Usage

Important: Switch keras backend to Theano and disable GPU execution (GPU memory is too limited for some of the experiments). GPU speedup for sparse operations is not that essential, so running this model on CPU will still be quite fast.

To replicate the experiments from our paper [1], first run (for AIFB):

```
python prepare_dataset.py -d aifb
```


Afterwards, train the model with:

```
python train.py -d aifb --bases 0 --hidden 16 --l2norm 0. --testing
```


Note that Theano performs an expensive compilation step the first time a computational graph is executed. This can take several minutes to complete.

For the MUTAG dataset, run:

```
python prepare_dataset.py -d mutag
python train.py -d mutag --bases 30 --hidden 16 --l2norm 5e-4 --testing
```

For BGS, run:

```
python prepare_dataset.py -d bgs
python train.py -d bgs --bases 40 --hidden 16 --l2norm 5e-4 --testing
```

For AM, run:

```
python prepare_dataset.py -d am
python train.py -d am --bases 40 --hidden 10 --l2norm 5e-4 --testing
```

Note: Results depend on random seed and will vary between re-runs.

## Setting keras backend to Theano

Create a file `~/.keras/keras.json` with the contents:

```
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

## Enforcing CPU execution


You can enforce execution on CPU by hiding all GPU resources:
```
CUDA_VISIBLE_DEVICES= python train.py -d aifb --bases 0 --hidden 16 --l2norm 0. --testing
```


## References

[1] M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, M. Welling, [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103), 2017


## Cite 

Please cite our paper if you use this code in your own work:

```
@article{schlichtkrull2017modeling,
  title={Modeling Relational Data with Graph Convolutional Networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N and Bloem, Peter and Berg, Rianne van den and Titov, Ivan and Welling, Max},
  journal={arXiv preprint arXiv:1703.06103},
  year={2017}
}
```