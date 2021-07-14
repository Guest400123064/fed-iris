# __Federated Iris Classifier: Prototype__

## __Overview__

This is an prototype _Federated Machine Learning_ application, specifically a simple _Iris Classifier_ (the classical iris dataset), based on primarily `torch` and `flwr`. Moreover, this is an experimental sub-project preparing for __*Fed-KGQA System*__, or __*Federated Question-Answering System over Knowledge Graph*__. The extended abstract can be found [NOT AVAILABLE ON _arXiv_ YET](). This repo implements the _Federated Answer Selector_ module mentioned in the extended abstract.

---
## __Pre-req__

Before getting started with this repo, a quick overview of _Federated Learning_ and the `flwr` package documentation if __HIGHLY RECOMMENDED__. The `flwr` official website, [click here](https://flower.dev/).

---
## __The General Architecture__

The application adopts the canonical _Server-Clients_ architecture, where:
* the clients are the training workers that:
  * updates their own models __locally and independently__
* the server is responsible for:
  * __collecting__ locally updated models
  * __aggregating__ the parameters/weights
  * __re-distribute__ the new global model back to the clients

Each one of the two sub-modules will be discussed in detail in their own `README`. In short, both the _Client_ module (under the `client` folder) and the _Server_ module (under the `server` folder) can be viewed as independent applications that can run individually.

---
## __Folder Structure__

Before running the code, I would like to introduce how I organized the various files. The folder structures of both the server and client sub-modules are quite similar.

### __Server Module__
```
root
|-- server
    |-- bin      # entry to the CLI
    |-- log      # training log
    |-- models   # cached model parameters
    |-- src      # python source code; called from the bin folder
        |-- __init__.py    # where the CLI is defined
```

### __Client Module__
```
root
|-- client
    |-- bin      # same as above
    |-- config   # local settings (not used yet; reserved)
    |-- data     # local training/validation data warehouse
    |-- models   # cached model parameters, for local model serving
    |-- src      # same as above
        |-- __init__.py    # where the CLI is defined, using components defined in the core module
        |-- core           # discussed in client/README.md
```

---
## __Quick Start__

In the future I may package the source code into (publication-ready) independent applications. For now I assume that the `conda` environment is properly setup or all the dependencies are installed. 

### __Server__

The server only serves as a parameter aggregator and currently takes only one command line argument `address`. If not provided, it starts a local server. 

```
# To start the server
#   assuming inside root directory
python ./server/bin/main --address '127.0.0.1:8080'

# For help
python ./server/bin/main --help
```

### __Client__

The client has two modes available: `infer` and `train`. The inference mode can operate independently:

```
python ./client/bin/main infer
```

and the expected interface is:

```
[ INFO ] :: entering inference mode
[ INFO ] :: press < ENTER > to make prediction; < Q > to exit: 
[ READ ] :: |-sep_len >>> 1.1
[ READ ] :: |-sep_wid >>> 2
[ READ ] :: |-pet_len >>> 3
[ READ ] :: |-pet_wid >>> 4
[ INFO ] :: it might be a < Iris-virginica >
```

To train the model we connect to the running server. Currently, purely local training is not supported. The client must connect to the server for federated training.

```
# Same as above, providing the address of the SERVER
#   by default, it tries to connect a local server
python ./client/bin/main train --address '127.0.0.1:8080'
```

__IMPORTANT:__
* Training won't start until at least __TWO__ clients are connected. However, this can be changed by configuring `flwr` [strategy](https://flower.dev/docs/strategies.html).
* After training, the new global model __IS NOT SAVED BY THE CLIENT__. Instead, the new model is stored under `./server/models` as an `.npz` file. So, for inference, please __manually__ copy the `model_params.npz` file to the `./client/models` folder.

---
## __Future Works__

Though workable, this prototype is still very "sketchy." Further developments largely fall into two broad categories:
* system optimization: __software engineering__
* more complex model: __question-answering model support__

### __The Software Part__
* Basic Requirements
  * error checking and handling
  * logging system
  * testing (robustness)
  * hard-code problem
  * etc.
* Design Problems
  * discussed independently in the sub-modules

### __The Model Support Part__

The iris classifier is a very simple example of _Softmax Regression_ model. But a QA model can be much more complex. One possible step further is to implement a _Text Similarity Matching_ model based on the [WikiQA](https://aclanthology.org/D15-1237/) dataset. In short, the format of training samples is _(Question, Bag-of-Answers)_ pairs. Thus, the task is to __rank the candidate answers based on its pertinency with the given question__, which is very similar to the _Answer Selector_ module mentioned in the extended abstract at the very beginning.