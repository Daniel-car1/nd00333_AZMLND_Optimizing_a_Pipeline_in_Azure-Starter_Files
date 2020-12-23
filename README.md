# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
The provided bankmarketing dataset is an open source csv file which has to be analysed by a Scikit-learn Hyperdrive Pipeline and an AutoML approach to predict weather a client subscribes a term deposit or not.
![overview](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/overview.PNG)
**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best perorming model
| Run type        | Algo.           | Accuracy  | Duration |
| ------------- |:-------------:| -----:| ----------:|
| Hyperdrive      | - | 0.915 | 46s |
| AutomatedML      | VotingEnsemble      |   0.916 | 1m19s |



## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
