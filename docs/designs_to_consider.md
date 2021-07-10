# Overall Architecture

## Client Side

* Client Architecture
  * Data-loader
    * Dataset (batch processing)
    * Real-time inference data-loader class?
  * Model
    * Sub-model initialization
  * Loss function class
  * Optimizer
* Client initialization
  * Train mode
  * Eval mode
    * Local data labelling and collection
* Logging
* Online training
  * Parameter exchange interface
  * Update default configuration on receive instructions from the server
    * Re-init?
* Model saving and loading