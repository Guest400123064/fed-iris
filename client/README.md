# __Federated Iris Classifier__

## __Overview__

As mentioned before, the classifier application has two modes, namely the _training mode_ and the _inference mode_. Both of these two modes share some modules in common:

* a data loader
* a classifier (the model)

Therefore we can extract these common modules and create separate classes for each one of them. Then, on a higher level, we utilize an _agent_ to encapsulate all the low-level details while only exposing the _APIs_.  

This is not a novel idea. A _repeatable_ machine learning project structure is thoroughly described in [PyTorch-Project-Template.](https://github.com/moemen95/PyTorch-Project-Template#tutorials) And I __HIGHLY RECOMMEND__ reading through their repo before preceding.

---
## __Software Architecture__

---
## __Design Problems to Consider__