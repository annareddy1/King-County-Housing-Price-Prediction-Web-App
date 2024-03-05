# King-County-Housing-Price-Prediction-Web-App
This project focuses on understanding the crucial features that influence house sales prices in King County, WA. By utilizing a dataset from Kaggle containing information about houses in the area, the goal is to create a web application that aids real estate developers, individual buyers, and financial institutions in making informed decisions. The app provides insights into the best areas in King County for developing new apartment buildings or making property purchases.

### Overview
Within this repository, you'll find an extensive project on machine learning and data visualization for King County housing data. The project's primary objective is to predict housing prices by analyzing various features such as square footage, number of bedrooms and bathrooms, house condition, and more. The process includes exploratory data analysis (EDA) and constructing statistical machine learning models using a range of supervised and unsupervised learning techniques.

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
- A concise report is available in HTML format: king_county_markdown.html.
- The same content is also provided in PDF format: king_county_markdown_report.pdf.

### Plots:
- Sample plots are included in the repository, showing distributions and relationships.
- Plots include maps, correlation plots, distributions, and relationships between variables.

### Additional Notes:
- Credit to "An Introduction to Statistical Learning" and "R Graphics Cookbook" for guidance.
- Various supervised and unsupervised learning methods used for regression and classification.
- The inclusion of classification and unsupervised learning methods serves as a demonstration of their potential applications.

### Conclusion
The King County Housing Price Prediction Web App project delves into the intricacies of housing prices in the area, offering valuable insights for developers, buyers, and financial institutions. Through thorough analysis, including exploratory data visualization and a range of machine learning models, the project aims to empower stakeholders with the information needed to make informed decisions regarding property development and investment in King County. The combination of supervised and unsupervised learning techniques allows for a comprehensive understanding of the dataset, providing a robust foundation for the predictive model within the web application.


