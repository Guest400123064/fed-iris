# __Federated Iris Classifier__

## __Overview__

As mentioned before, the classifier application has two modes, namely the _training mode_ and the _inference mode_. Both of these two modes share some modules in common:

* a data loader
* a classifier (the model)

Therefore we can extract these common modules and create separate classes for each one of them. Then, on a higher level, we utilize an _agent_ to encapsulate all the low-level details while only exposing the _APIs_.  

This is not a novel idea. A _repeatable_ machine learning project structure is thoroughly described in [PyTorch-Project-Template.](https://github.com/moemen95/PyTorch-Project-Template#tutorials) And I __HIGHLY RECOMMEND__ reading through their repo before preceding.

---

## __Software Architecture__

In short, there are three layers of abstraction in building an agent:

* Data Loader
  * User input reader
  * Training dataset (batch) loader
* Model
  * Implements training and prediction methods
* External API
  * For training
  * For inference

With this architecture in mind, the code base is relative straight forward.

---

## __Design Problems to Consider__

### __A Unified Agent__

Currently there are two separate agents for training and inference respectively. This is because:

* training agents requires a _loss function component_ and a _optimizer component_ which are useless during inference stage.
* model training generally requires multiple input data streams as training and validation samples respectively while there is only one data pipeline during the prediction stage.

Then, is there a better way to design the __initializer of an agent__ so that we don't need to classes?

### __Integrating Teaching Mode into the Agent Class__

An important functionality of a federated machine learning application is being able to let users contribute new data, to "teach" the model. And currently such feature is not available.

### __Pull Latest Model from the Parameter Server__

Although the server will distribute the latest model to connected clients during training stage, the model cannot request a model from the server during inference phase.

### __Support for Sub-model-only Training__

The current classifier is simple: a single linear layer. However, if we have a humongous model with multiple sub-models and millions of parameters, we may want to adopt a __update-by-parts__ scheme, that is, we update and upload only the parameters of that specific sub-module. This feature is not available yet.
