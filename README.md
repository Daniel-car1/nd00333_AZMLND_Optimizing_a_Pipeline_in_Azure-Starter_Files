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
| **Hyperdrive**      | **LogisticRegression (SKlearn)** | 0.914 | 54s |
| **AutomatedML**      | **VotingEnsemble**      |   0.916 | 1m25s |

## Scikit-learn Pipeline
The Scikit-learn Pipeline followes the CRIP-DM stages and tasks like importing data to obtain a editable dataset, cleaning and filtering data, tuning the Hyperparameters regularization strength (--C) and maximal number of iterations (--max_iter) using Hyperdrive and classify using *logistic regression*.
* '--C', a poitive value, describes the inverse of the regularization strength, smaller values specify stronger regularization.
* '--mat_iter', describes the maximum number of iterations, taken from the solvers to converge.

*RandomParameterSampling*, a random parameter sampler which supports disrcrete like uniform and continuous like choice hyperameters. Hyperparamerters are randomly selected from the defined search space and early termination of low-performance runs is supported. Compared to GridParameterSampling, an other parameter sampler which supports only descrete hyperparameters. For more improvements, random sampling can be used for inital hyperparameter search and then refine it using Bayesion sampling, which picks samples based on how previous samples performed, so that new samples improve the primary metric. 

Using the *BanditPolicy* - an early stopping policy - to terminate badly performing runs. This algorithm can be adusted by the parameters evaluation interval and slack factor. A run that does not fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

* slack_factor: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.
* evaluation_interval: An optional parameter, which describes the frequency for applying the policy.

**Hyperdrive achieved an accuracy of 0.9144 with the best fitted parameters for the regularization strength '--C' of 0.7386 and the maximal number of iterations '--max_iter' of 200.**

![hyperdrive](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/Hyperdrive.PNG)

## AutoML
*AutoML* is a machine learning technology to train and tune a model using the target metric for a dataset to find the best fitting model. Due to the task, the AutoML configuration provides the classification task and accuracy as primary metric to compare the best AutoML model with the Scikit-learn Pipeline. Several classification algorithms like LogisticRegression, RandomForest etc. were tested. **The best fitting classification algorithm VotingEnsemble reached an accuracy of 0.9157 after an duration of 1m25s.**

![AutoML](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/AutoML_algos.PNG)

*VotingEnsemble* is a weighted superposition of classification algorithms, called ensembled_algorithms ('XGBoostClassifier','LightGBM' and 'RandomForest') with the corresponding ensemble_weights (0.0833, 0.5, 0.1666, etc.). More detailed inforamtion can be obtained in [VotingEnsemble_Raw JSON](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/EnsembleV_RawJSON.json) and in the following slide. 

![AutoML](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/RawJSON_Ensemble_AutoML.PNG)

The explained best AutoML generated model generated the following parameters (marked with blue in the next image):
* min_sample_leaf=0.01; helps to avoid overfitting.
* min_sample_split=0.01; can create arbitrary small leaves, while min_sample_leaf guarantees a minimum number of samples in a leave.
* min_weight_fraction_leaf=0.0;
* n_estimators=25;
* n_jobs=1;

![AutoML](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/AutoML_parameter2.PNG)

## Pipeline comparison
The accuracy obtained by the Scikit-learn pipeline with Hyperdrive hyperparameter tuning is 0.914. Whereas, the accuracy obtained by the best AutoMl model VotingEnsemble reaches 0.916, both values are quite similar. Deviatins result from the different architectures, the Scikit-learn pipline uses the logistic regression classification algorithm, whereas AutoML tests a pool of different classification algorithms.

## Future work
* Cleaning the data more precise
* Improving the parameter sampler
* Update the clean_data function to obtain pandas datasets and tablular dataset
* Try out BayesianParameter Sampling Technique which intelligently picks the next sample of hyperparameters, based on how the previous samples performed, such that the new sample improves the reported primary metric.

## Proof of cluster clean up
Deleting the cluster in the notebook.

![AutoML](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/Delete_code.PNG)

![AutoML](https://github.com/Daniel-car1/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/docs/Delete.PNG)

## Sources
* https://www.udacity.com/
* https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py
* https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters
* https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets?WT.mc_id=AI-MVP-5003930#create-a-dataset-from-pandas-dataframe
* https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-auto-train-models
* https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
* https://stackoverflow.com/

