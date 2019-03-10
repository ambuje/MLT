
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
def dataset_import(file_path,X_row1,X_row2,X_col1,X_col2,y_row1,y_row2, y_col):
  dataset = pd.read_csv(file_path)
  X = dataset.iloc[X_row1:X_row2, X_col1:X_col2].values #Independent Variable
  y = dataset.iloc[y_row1:y_row2, y_col].values  #Dependent Variable
  

# Feature Scaling
def feature_scaling1():
  from sklearn.preprocessing import StandardScaler
  sc_X = StandardScaler()
  X_train = sc_X.fit_transform(X_train)
  X_test = sc_X.transform(X_test)
  sc_y = StandardScaler()
  y_train = sc_y.fit_transform(y_train)

# Simple Linear Regression
def slr():
  # Importing the dataset
  #dataset = pd.read_csv('C:\\Users\\harsh\\Downloads\\Sem III Project ML\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 4 - Simple Linear Regression\\Simple_Linear_Regression\\Salary_Data.csv')
  dataset = pd.read_csv('fin_ind.csv')
  X = dataset.iloc[:, 0].values
  y = dataset.iloc[:, 1].values

  # Splitting the dataset into the Training set and Test set
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

  # Fitting Simple Linear Regression to the Training set
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(X_train, y_train)

  # Predicting the Test set results
  y_pred = regressor.predict(X_test)
  #a=int(input("Enter year of experience"))
  
  #y_pred1=regressor.predict(a)
  #print("Year of Experience is" ,y_pred1)

  # Visualising the Training set results
  plt.scatter(X_train, y_train, color = 'red')
  plt.plot(X_train, regressor.predict(X_train), color = 'blue')
  plt.title('Salary vs Experience (Training set)')
  plt.xlabel('Years of Experience')
  plt.ylabel('Salary')
  plt.show()
  #plt.switch_backend('qt4agg')


  # Visualising the Test set results
  plt.scatter(X_test, y_test, color = 'red')
  plt.plot(X_train, regressor.predict(X_train), color = 'blue')
  plt.title('Salary vs Experience (Test set)')
  plt.xlabel('Years of Experience')
  plt.ylabel('Salary')
  plt.show()
  
# Multiple Linear Regression
def mlr():
  # Importing the dataset
  dataset = pd.read_csv('50_Startups.csv')
  X = dataset.iloc[:, :3].values
  y = dataset.iloc[:, 4].values

  # Encoding categorical data
  from sklearn.preprocessing import LabelEncoder, OneHotEncoder
  labelencoder = LabelEncoder()
  X[:, 3] = labelencoder.fit_transform(X[:, 3])
  onehotencoder = OneHotEncoder(categorical_features = [3])
  X = onehotencoder.fit_transform(X).toarray()

  # Avoiding the Dummy Variable Trap
  X = X[:, 1:]

  # Splitting the dataset into the Training set and Test set
  from sklearn.cross_validation import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

  # Fitting Multiple Linear Regression to the Training set
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(X_train, y_train)

  # Predicting the Test set results
  y_pred = regressor.predict(X_test)
  
  # Visualising the Training set results
  plt.scatter(X_train[:,0], y_train, color = 'red')
  #plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
  plt.title('MLR')
  plt.xlabel('Independent Variable 1')
  plt.ylabel('Dependent Variable')
  plt.show()
  #plt.switch_backend('qt4agg')
  
  plt.scatter(X_train[:,1], y_train, color = 'red')
  #plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
  plt.title('MLR')
  plt.xlabel('Independent Variable 2')
  plt.ylabel('Dependent Variable')
  plt.show()
  
  plt.scatter(X_train[:,2], y_train, color = 'red')
  #plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
  plt.title('MLR')
  plt.xlabel('Independent Variable 3')
  plt.ylabel('Dependent Variable')
  plt.show()


  # Visualising the Test set results
  plt.scatter(X_test[:,0], y_test, color = 'red')
  #plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
  plt.title('MLR')
  plt.xlabel('Independent Variable 1')
  plt.ylabel('Dependent Variable')
  plt.show()
  #plt.switch_backend('qt4agg')
  
  plt.scatter(X_test[:,1], y_test, color = 'red')
  #plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
  plt.title('MLR')
  plt.xlabel('Independent Variable 2')
  plt.ylabel('Dependent Variable')
  plt.show()
  
  plt.scatter(X_test[:,2], y_test, color = 'red')
  #plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
  plt.title('MLR')
  plt.xlabel('Independent Variable 3')
  plt.ylabel('Dependent Variable')
  plt.show()
  
# Polynomial Regression

def pr():
  # Importing the dataset
  dataset = pd.read_csv('Position_Salaries.csv')
  X = dataset.iloc[:, 1:2].values
  y = dataset.iloc[:, 2].values

  # Fitting Linear Regression to the dataset
  from sklearn.linear_model import LinearRegression
  lin_reg = LinearRegression()
  lin_reg.fit(X, y)

  # Fitting Polynomial Regression to the dataset
  from sklearn.preprocessing import PolynomialFeatures
  poly_reg = PolynomialFeatures(degree = 4)
  X_poly = poly_reg.fit_transform(X)
  poly_reg.fit(X_poly, y)
  lin_reg_2 = LinearRegression()
  lin_reg_2.fit(X_poly, y)

  # Visualising the Linear Regression results
  plt.scatter(X, y, color = 'red')
  plt.plot(X, lin_reg.predict(X), color = 'blue')
  plt.title('Truth or Bluff (Linear Regression)')
  plt.xlabel('Position level')
  plt.ylabel('Salary')
  plt.show()

  # Visualising the Polynomial Regression results
  plt.scatter(X, y, color = 'red')
  plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
  plt.title('Truth or Bluff (Polynomial Regression)')
  plt.xlabel('Position level')
  plt.ylabel('Salary')
  plt.show()

  # Visualising the Polynomial Regression results (for higher resolution and smoother curve)
  X_grid = np.arange(min(X), max(X), 0.1)
  X_grid = X_grid.reshape((len(X_grid), 1))
  plt.scatter(X, y, color = 'red')
  plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
  plt.title('Truth or Bluff (Polynomial Regression)')
  plt.xlabel('Position level')
  plt.ylabel('Salary')
  plt.show()

  # Predicting a new result with Linear Regression
  lin_reg.predict(6.5)

  # Predicting a new result with Polynomial Regression
  lin_reg_2.predict(poly_reg.fit_transform(6.5))
  
# SVR
"""
def svr():
  # Importing the dataset
  dataset = pd.read_csv('Position_Salaries.csv')
  X = dataset.iloc[:, 1:2].values
  y = dataset.iloc[:, 2].values

  # Feature Scaling
  from sklearn.preprocessing import StandardScaler
  sc_X = StandardScaler()
  sc_y = StandardScaler()
  X = sc_X.fit_transform(X)
  y = sc_y.fit_transform(y)

  # Fitting SVR to the dataset
  from sklearn.svm import SVR
  regressor = SVR(kernel = 'rbf')
  regressor.fit(X, y)

  # Predicting a new result
  y_pred = regressor.predict([[6.5]])
  y_pred = sc_y.inverse_transform(y_pred)

  # Visualising the SVR results
  plt.scatter(X, y, color = 'red')
  plt.plot(X, regressor.predict(X), color = 'blue')
  plt.title('Truth or Bluff (SVR)')
  plt.xlabel('Position level')
  plt.ylabel('Salary')
  plt.show()

  # Visualising the SVR results (for higher resolution and smoother curve)
  X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
  X_grid = X_grid.reshape((len(X_grid), 1))
  plt.scatter(X, y, color = 'red')
  plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
  plt.title('Truth or Bluff (SVR)')
  plt.xlabel('Position level')
  plt.ylabel('Salary')
  plt.show()
  """
  
# Decision Tree Regression
def dtr():
  # Importing the dataset
  dataset = pd.read_csv('Position_Salaries.csv')
  X = dataset.iloc[:, 1:2].values
  y = dataset.iloc[:, 2].values

  # Fitting Decision Tree Regression to the dataset
  from sklearn.tree import DecisionTreeRegressor
  regressor = DecisionTreeRegressor(random_state = 0)
  regressor.fit(X, y)

  # Predicting a new result
  y_pred = regressor.predict(6.5)

  # Visualising the Decision Tree Regression results (higher resolution)
  X_grid = np.arange(min(X), max(X), 0.01)
  X_grid = X_grid.reshape((len(X_grid), 1))
  plt.scatter(X, y, color = 'red')
  plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
  plt.title('Truth or Bluff (Decision Tree Regression)')
  plt.xlabel('Position level')
  plt.ylabel('Salary')
  plt.show()
  
# Random Forest Regression
def rfr():
  # Importing the dataset
  dataset = pd.read_csv('Position_Salaries.csv')
  X = dataset.iloc[:, 1:2].values
  y = dataset.iloc[:, 2].values

  # Fitting Random Forest Regression to the dataset
  from sklearn.ensemble import RandomForestRegressor
  regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
  regressor.fit(X, y)

  # Predicting a new result
  y_pred = regressor.predict(6.5)

  # Visualising the Random Forest Regression results (higher resolution)
  X_grid = np.arange(min(X), max(X), 0.01)
  X_grid = X_grid.reshape((len(X_grid), 1))
  plt.scatter(X, y, color = 'red')
  plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
  plt.title('Truth or Bluff (Random Forest Regression)')
  plt.xlabel('Position level')
  plt.ylabel('Salary')
  plt.show()
  

slr()
mlr()
pr()
dtr()
rfr() 