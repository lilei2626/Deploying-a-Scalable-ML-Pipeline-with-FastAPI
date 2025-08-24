# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Logistic Regression classifier trained on the UCI Adult Census dataset (also known as the Census Income dataset).
The implementation uses scikit-learn’s LogisticRegression with max_iter=1000 and class_weight="balanced". The model was developed inside a Python 3.10 virtual environment using pandas, scikit-learn, and joblib.

## Intended Use
The model is intended to predict whether a person earns more than $50K per year based on demographic and employment attributes.
It may be used as an educational example for building ML pipelines, deploying models, and documenting model behavior.
This model should not be used in production systems that affect people’s lives without extensive fairness, bias, and privacy audits.

## Training Data
The model was trained on the census.csv dataset provided in the starter repository.
The dataset originates from the 1994 U.S. Census Bureau database and contains features such as:

Age,
Workclass,
Education,
Marital-status,
Occupation,
Relationship,
Race,
Sex,
Native-country,

The target label is salary, which is binary: <=50K or >50K.

## Evaluation Data
The dataset was split into 80% training and 20% testing, stratified by the target label to preserve class distribution.
The test set was used to evaluate overall model performance and performance on categorical slices (e.g., by education level, race, sex).

## Metrics
_Please include the metrics used and your model's performance on those metrics._
We evaluated the model using the following metrics:
Precision - 0.5507,
Recall - 0.8272,
F1 score (F-beta, beta=1) - 0.6612

We also evaluated performance on slices of categorical features (values of education, workclass, race, sex).
The metrics for each slice are logged in slice_output.txt.

## Ethical Considerations
The Census dataset includes sensitive features such as race and sex.
Using this model in real-world decision making could reinforce or increase existing social biases.
Predictions should not be used for employment, credit, or housing decisions without fairness analysis and bias mitigation strategies.
Additionally, the dataset is dated (1994), and relationships between demographic attributes and income may not generalize to current populations.

## Caveats and Recommendations
This model is for educational purposes only.
The dataset is imbalanced (<=50K is much more common than >50K), which impacts performance.
Future improvements could include:
Hyperparameter tuning or trying other algorithms (e.g., Random Forest, Gradient Boosting).
Using techniques to mitigate bias across sensitive attributes.
Normalizing continuous features and exploring feature engineering.
Cross-validation instead of a single train/test split.