# King-County-Housing-Price-Prediction-Web-App
This project enables figuring out the key features that determine the sales price of houses. The resulting Web App helps real estate developers, individual buyers, and banks seek the best area in King County to develop new apartment buildings or make purchases.

### Overview
This repository contains a comprehensive project on machine learning and data visualization for King County housing data. The project aims to predict housing prices using a dataset of houses located in King County, WA, obtained from Kaggle. Each observation represents a house in the county, with features such as square footage, number of bedrooms and bathrooms, condition, and more. The project includes exploratory data analysis (EDA) and the construction of statistical machine learning models using various supervised and unsupervised learning methods.

### Data Files:

Features in the Dataset:
- price: Price of each house.
- sqrt_price: Square root of the price.
- price2: Dichotomous price variable (0 for houses <$600,000, 1 for houses <=$600,000).
- sqft_living, sqft_lot, sqft_above, sqft_basement, sqft_living15, sqft_lot15: Square footage features.
  bedrooms, bathrooms, floors, waterfront, condition, grade: House features.
- yr_built, zipcode, lat, long: Location and year built features.

### Scripts and Models:

1. EDA and Visualizations (1 - EDA and visualizations.R):

Map of houses in King County.
Summary statistics and plots of variables.
Correlations among variables.
Relationship between sqrt_price (outcome) and predictors.

2. Linear Regression (2 - linear regression.R):

Simple and multiple linear regression.
Polynomial regression with cross-validation.
Plotting linear polynomial regression.

3. Subset Selection (3 - subset selection.R):

Best subset selection.
Forward and backward stepwise selection.
Model selection using validation set and cross-validation.

4. Ridge Regression and Lasso (4 - ridge regression and lasso.R):

Ridge regression with and without cross-validation.
Lasso regression with and without cross-validation.

5. PCR and PLS (5 - PCR & PLS.R):

Principal components regression (PCR) with cross-validation.
Partial least squares (PLS) with cross-validation.

6. Logistic Regression (6 - logistic regression.R):

Logistic regression and logistic polynomial regression with cross-validation.
Plotting logistic polynomial regression.

7. LDA, QDA, and KNN (7 - LDA, QDA, and KNN.R):

Linear discriminant analysis (LDA).
Quadratic discriminant analysis (QDA).
K-nearest neighbors (KNN) for categorization.

8. Decision Trees (8 - decision trees.R):

Classification and regression trees.
Bagging, random forests, and boosting.

9. Support Vector Machines (9 - support vector machines.R):

Support vector classifiers and machines for classification.

10. Unsupervised Learning (10 - unsupervised learning.R):

Principal components analysis (PCA).
K-means and hierarchical clustering.

### Report:
- A short report is available in HTML format: king_county_markdown.html.
- The same content is available in PDF format: king_county_markdown_report.pdf.

### Plots:
- Sample plots are included in the repository, showing distributions and relationships.
- Plots include maps, correlation plots, distributions, and relationships between variables.

### Additional Notes:
- Credit to "An Introduction to Statistical Learning" and "R Graphics Cookbook" for guidance.
- Various supervised and unsupervised learning methods used for regression and classification.
- Explanation about the inclusion of classification and unsupervised learning for demonstration purposes.

This repository seems to be a comprehensive project focusing on exploring and modeling King County housing data using various machine learning techniques.
