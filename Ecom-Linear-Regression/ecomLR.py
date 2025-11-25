import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load e-commerce dataset
print("Loading e-commerce dataset...")
data = pd.read_csv('Ecommerce Customers')
print("Dataset loaded successfully.")
# print(data.head())
print(data.info())
# print(data.describe())


#EDA and cleaning the Data
g1 = sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=data)
g1.savefig("jointplot_time_website_vs_yearly_spent.png")

g2 = sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=data)
g2.savefig("jointplot_time_app_vs_yearly_spent.png")

g3 = sns.pairplot(data=data, plot_kws={'alpha':0.5})
g3.savefig("pairplot_data.png")

g4 = sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=data)
g4.savefig("lmplot_length_membership_vs_yearly_spent.png")

#Training and Testing Data
from sklearn.model_selection import train_test_split

X = data[['Avg. Session Length','Time on App','Time on Website', 'Length of Membership']]
y = data['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

#Training the Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train,y_train)
print("Model trained successfully.")

# print(lm.coef_)

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['coefficients'])
print(cdf)


#Predicting Test Data
predictions = lm.predict(X_test)
# print(predictions)
g5 = sns.scatterplot(x=predictions,y=y_test)
plt.xlabel("Predicted Values")
# plt.ylabel("Y Test Values")
plt.title("Predicted vs Actual Yearly Amount Spent")
g5.figure.savefig("predicted_vs_actual.png")


#Evaluating the Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test,predictions)
rmse = root_mean_squared_error(y_test, predictions)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)


#Residuals
residuals = y_test - predictions
# g6 = sns.histplot(residuals, bins=30, kde=True)
g7 = sns.displot(residuals,bins=30, kde=True)
# g6.figure.savefig("residuals_histogram.png")
g7.savefig("residuals_displot.png")


import pylab as pl
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=pl)
pl.title("Q-Q Plot of Residuals")
pl.savefig("qq_plot_residuals.png")



