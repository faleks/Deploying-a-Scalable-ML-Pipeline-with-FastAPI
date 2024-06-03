# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was create by Fatima Aleksander for the ML DevOps course from Udacity. It uses RandomForestClassifier to predict income level from a 1994 Census dataset.

## Intended Use
This model should be used to identify whether an individual's income exceeds $50K per year.

## Training Data
Information on the dataset can be found here: https://archive.ics.uci.edu/ml/datasets/census+income 

## Evaluation Data
The model is evaluated on a test dataset separate from the training data. The original dataset was preprocessed and split.

## Metrics
The model's performance is evaluated using precision, recall, and fbeta scores. 
Performance results: Precision: 0.7089 | Recall: 0.6315 | F1: 0.6680

## Ethical Considerations
The census data may contain biases or inaccuracies based on various demographics that could influence the model's predictions.

## Caveats and Recommendations
The model's accuracy is limited by the quality of the data. The dataset is from 1994 census data which may not accurately represent different populations or scenarios for some predictions.