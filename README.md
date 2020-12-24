# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The provided bankmarketing dataset is an open source csv file which has to be analysed by a Scikit-learn Hyperdrive Pipeline and an AutoML approach to predict weather a client subscribes a term deposit or not.
![overview](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/overview.PNG)
The best perorming model was the VotingEnsemble, an AutomatedML approach which gave an accuracy of 0.916. [saved VotingEnsemble model](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/AutoMLb55d7c87225.zip)
| Run type        | Algorithm           | Accuracy  | Duration |
| ------------- |:-------------:| -----:| ----------:|
| Hyperdrive      | LinearRegression (SKlearn) | 0.914 | 54s |
| AutomatedML      | VotingEnsemble      |   0.916 | 1m25s |

## Scikit-learn Pipeline
The Scikit-learn Pipeline followes the CRIP-DM stages and tasks like importing data to obtain a editable dataset, cleaning and filtering data, tuning the Hyperparameters regularization strength and maximal number of iterations using Hyperdrive and classify using linear regression.
RandomParameterSampling, a parameter sampler which supports disrcrete like uniform and continuous like choice hyperameters. Hyperparamerters are randomly selected from the defined search space.
Using the BanditPolicy - an early stopping policy - to terminate badly performing runs. This algorithm can be adusted by the parameters evaluation interval and slack factor. A run that does not fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.
* slack_factor: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.
* evaluation_interval: An optional parameter, which describes the frequency for applying the policy.
Hyperdrive achieved an accuracy of 0.9144 with the best fitted parameters for the regularization strength '--C' of 0.7386 and the maximal number of iterations '--max_iter' of 200.
![hyperdrive](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/Hyperdrive.PNG)

## AutoML
AutoML is a machine learning technology to train and tune a model using the target metric for a dataset to find the best fitting model. Due to the task, the AutoML configuration provides the classification task and accuracy as primary metric to compare the best AutoML model with the Scikit-learn Pipeline. Several classification algorithms like LogisticRegression, RandomForest etc. were tested. The best fitting classification algorithm VotingEnsemble reached an accuracy of 0.9157 after an duration of 1m25s.
![AutoML](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/AutoML_algos.PNG)
VotingEnsemble is a weighted superposition of classification algorithms, called ensembled_algorithms ('XGBoostClassifier','LightGBM' and 'RandomForest') with the corresponding ensemble_weights (0.0833, 0.5, 0.1666, etc.). More detailed inforamtion can be obtained in [VotingEnsemble_Raw JSON](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/EnsembleV_RawJSON.json) and in the following slide. 
![AutoML](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/RawJSON_Ensemble_AutoML.PNG)


## Pipeline comparison
The accuracy obtained by the Scikit-learn pipeline with Hyperdrive hyperparameter tuning is 0.914. Whereas, the accuracy obtained by the best AutoMl model VotingEnsemble reaches 0.916, both values are quite similar. Deviatins result from the different architectures, the Scikit-learn pipline uses the linear regression classification algorithm, whereas AutoML tests a pool of different classification algorithms.
The AutoML is a useful approach, because it tests different algorithms.

## Future work
* Cleaning the data more precise
* Improving the parameter sampler
* Update the clean_data function to obtain pandas datasets and tablular dataset
* Try out BayesianParameter Sampling Technique which intelligently picks the next sample of hyperparameters, based on how the previous samples performed, such that the new sample improves the reported primary metric.

## Proof of cluster clean up
![AutoML](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/Delete_code.PNG)
![AutoML](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/Delete.PNG)

## Sources
https://www.udacity.com/
