# 🏥 Medical Insurance Cost Prediction

A Machine Learning project that predicts estimated medical insurance charges based on demographic and lifestyle information.

The project uses Multiple Linear Regression and provides an interactive Streamlit web application where users can enter individual information and receive an estimated insurance cost.

---

## 📌 Project Overview

[Live Application / Demo](https://medical-insurance-cost-predicto.streamlit.app/)

Medical insurance charges can vary depending on factors such as age, BMI, smoking status, number of children, gender, and geographical region.

The goal of this project is to build a Machine Learning regression model that learns relationships between these features and historical insurance charges and then predicts an estimated insurance cost for a new individual.

---

## 🎯 Problem Statement

The objective is to predict the estimated medical insurance cost of an individual using:

- Age
- Gender
- BMI
- Number of Children
- Smoking Status
- Region

The target variable is:

`charges`

Since the target is a continuous numerical value, this is a **Regression** problem.

---

## 📊 Dataset

The project uses the **Medical Cost Personal Dataset**.

The original dataset contains **1,338 records and 7 columns**.

After removing one duplicate record during preprocessing, the final dataset contains:

**1,337 records × 7 columns**

### Features

| Feature | Description |
|---|---|
| age | Age of the individual |
| sex | Gender |
| bmi | Body Mass Index |
| children | Number of children/dependents |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance charges |

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed.

### Missing Values

Missing values were checked using:

```python
df.isnull().sum()
