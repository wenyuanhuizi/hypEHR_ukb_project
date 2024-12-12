# Predicting Cardiovascular Disease Among Type 2 Diabetes Patients: A Comparative Study of Baseline Models and HypEHR

This project investigates the prediction of cardiovascular diseases (CVD) among Type 2 Diabetes (T2D) patients. We conduct a comparative study between a state-of-the-art model, [HypEHR](https://pmc.ncbi.nlm.nih.gov/articles/PMC10283128/), and a suite of baseline models.

## Data

The data folder contains the following:

- Adjacency list: Represents the relationships between medical codes within patient visits.
- Node embeddings: Generated for processing the data into the various models.
## Baseline Models

The baseline folder includes implementations of the following models:

- Logistic Regression (LR)
- Support Vector Machine (SVM)
- Random Forest (RF)
- Feedforward Neural Network (FNN)
- Graph Neural Network (GNN)
## HypEHR Model

The hypEHR folder contains our modified and implemented version of the HypEHR model.

## Key Objectives

- Evaluate the performance of HypEHR in predicting CVD among T2D patients.
- Compare the performance of HypEHR against a set of established baseline models.
- Gain insights into the effectiveness of different model architectures for this specific medical prediction task.

