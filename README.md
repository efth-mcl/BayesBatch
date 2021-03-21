# Bayes & Batch 
### Increasing batch size during training process with the use of bayesian optimization.
This project based on academic project on Deep Learning course. The dataset which is used is FashionMNIST with \
the highest accuracy around 94%. The academic project created by 2 students where one is me. The project based on Keras \
of TesnorFlow 2.0.
#### Requirements
- [scikit-image](https://github.com/scikit-image/scikit-image)
- [tensorflow](https://github.com/tensorflow/tensorflow)
- [bayesian-optimization](https://github.com/fmfn/BayesianOptimization)
- [pandas_ml](https://github.com/pandas-ml/pandas-ml)
- [scikit-learn-ZCA](https://github.com/mwv/zca)

## Notebook pipeline
1. Train per preprocessing type. \
   preprocessing types:
   * No preprocessing
   * Standard division normalization
   * ZCA whitening
   * Histogram Equalization
    
    training types:
   * Simple Training: train with static hyperparameters (2 Stacks of CNNs ).
   * Batch increasing method: Increse batch size during training until stops with upper threshold.
   * Bayes Opt: Bayesian Optimization where hyperparameters changes in specific range, using Simple Training,
   * Bayes + Batch increasing method: Combine Bayesian Optimization & incrissing batch size.
   
   For range hyperparameter ranges see BayesBatch notebook. Using two hyperparameter objects one for simple training \
   and one for bayes opt.
   
2. Results: Show bar chart, and loss/accuracy curves. The best method is Bayes + Batch increasing method with test 
   score 94%.
