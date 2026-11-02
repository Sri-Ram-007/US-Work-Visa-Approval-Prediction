# US Work Visa Approval Prediction

A machine learning project that predicts whether a US employment-based visa application will be **Certified** or **Denied**, from details of the applicant, the job and the sponsoring company.

Team project for the Machine Learning Lab (24CS384), Siddhartha Academy of Higher Education, by **B. Priya Vardhan**, **J. Akhilandeshwar** and **K. Sri Ram**.

## Dataset

[EasyVisa on Kaggle](https://www.kaggle.com/datasets/moro23/easyvisa-dataset), originally from the Office of Foreign Labor Certification, US Department of Labor.

- 25,480 applications and 12 columns, with no missing values or duplicate rows
- 67% Certified and 33% Denied, so the classes are imbalanced

The dataset is not included in this repo. Download `EasyVisa.csv` from Kaggle and place it next to the notebook.

## Approach

1. **Exploration:** approval rates by education, experience, wage unit and continent.
2. **Cleaning:** drop the ID column and fix 33 negative employee counts.
3. **Feature engineering:** convert wages to a yearly figure, add company age, and log-transform skewed columns.
4. **Preparation:** binary, ordinal and one-hot encoding, an 80/20 stratified split, and scaling fitted on training data only.
5. **Modelling:** compare Logistic Regression, Random Forest and Gradient Boosting, then tune the Random Forest with GridSearchCV (3-fold cross-validation).
6. **Delivery:** save the preprocessing and model as one Joblib pipeline, and serve it through a Gradio app.

## Results

Measured on 5,096 test applications the models never saw:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 72.4% | 0.753 | 0.874 | 0.809 |
| Random Forest | 72.2% | 0.768 | 0.837 | 0.801 |
| Gradient Boosting | 74.1% | 0.771 | 0.871 | 0.818 |
| **Tuned Random Forest** | **74.1%** | **0.766** | **0.881** | **0.819** |

Always guessing "Certified" would already score about 67%, so F1 and recall were checked as well as accuracy. Education was the strongest factor, at about 40% of the final model's feature importance.

## Limitations

- It covers US work visas only, and the data records the applicant's continent rather than their country.
- Real decisions depend on things not in the data, such as documents and job details, which limits accuracy.
- Denials are harder to predict because there are half as many of them.

This is a student project. Its predictions are not legal or immigration advice.

## Run it

Open `US_Work_Visa_Approval_Prediction.ipynb` in Google Colab or Jupyter, add `EasyVisa.csv` to the same folder, and run the cells in order. The last cell starts the Gradio app.

```
pip install -r requirements.txt
```
