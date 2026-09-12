import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
data = pd.read_csv(
    r"carpriceproject/data/car.csv"
)
df=pd.DataFrame(data)
print(df)

# data cleaning 
print(df.info())
print(df.describe())
print(df.shape)
print(df.dtypes)
print(df.columns)
print(df.isnull().sum())
print(df['model'].value_counts())
print(df['brand'].value_counts())
print(df['fuelType'].value_counts())
print(df['transmission'].value_counts())

print(df.groupby(['brand','transmission'])['price'].agg(['count','mean','sum','max','min']))
print(df.groupby(['brand','transmission'])['price'].sum())

print(df.groupby(['brand','fuelType'])['price'].sum())
print(df[['mileage', 'mpg']].corr())
# graph
categorial=['brand','fuelType','transmission']
for col in categorial:
 plt.figure(figsize=(8,4))
 sns.histplot(data=df,x=df[col])
 plt.show()

sns.scatterplot(data=df, x='year', y='price')
plt.show()

sns.scatterplot(data=df, x='mileage', y='price')
plt.show()

sns.boxplot(data=df, x='fuelType', y='price')
plt.show()

sns.boxplot(data=df, x='transmission', y='price')
plt.show()

sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

# create x and y
df_encode=pd.get_dummies(df,drop_first=True)
X=df_encode.drop(['price'],axis=1)
y=df_encode['price']


# test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# standard scale
scalar=StandardScaler()
X_train_scaled=scalar.fit_transform(X_train)
X_test_scaled=scalar.transform(X_test)

# making model
model_lr=LinearRegression()
model_lr.fit(X_train,y_train)
y_pred_lr=model_lr.predict(X_test)



# decision tree
model_DT=DecisionTreeRegressor(random_state=42)
model_DT.fit(X_train_scaled,y_train)
y_pred_DT=model_DT.predict(X_test_scaled)


# SVM
model_SVM=SVR(kernel='rbf')
model_SVM.fit(X_train_scaled,y_train)
y_pred_SVM=model_SVM.predict(X_test_scaled)



# KNN regressor
model_KNN=KNeighborsRegressor(n_neighbors=5)
model_KNN.fit(X_train_scaled,y_train)
y_pred_KNN=model_KNN.predict(X_test_scaled)


# gradient boosting 
model_GD=GradientBoostingRegressor(random_state=42)
model_GD.fit(X_train,y_train)
y_pred_GD=model_GD.predict(X_test)


# Random forest model
model_RF=RandomForestRegressor(n_estimators=200,random_state=42)
model_RF.fit(X_train,y_train)
y_pred_RF=model_RF.predict(X_test)


# # Model Comparison

results = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Decision Tree',
        'SVR',
        'KNN',
        'Gradient Boosting',
        'Random Forest'
    ],

    'MAE': [
        mean_absolute_error(y_test, y_pred_lr),
        mean_absolute_error(y_test, y_pred_DT),
        mean_absolute_error(y_test, y_pred_SVM),
        mean_absolute_error(y_test, y_pred_KNN),
        mean_absolute_error(y_test, y_pred_GD),
        mean_absolute_error(y_test, y_pred_RF)
    ],

    'MSE': [
        mean_squared_error(y_test, y_pred_lr),
        mean_squared_error(y_test, y_pred_DT),
        mean_squared_error(y_test, y_pred_SVM),
        mean_squared_error(y_test, y_pred_KNN),
        mean_squared_error(y_test, y_pred_GD),
        mean_squared_error(y_test, y_pred_RF)
    ],

    'RMSE': [
        np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        np.sqrt(mean_squared_error(y_test, y_pred_DT)),
        np.sqrt(mean_squared_error(y_test, y_pred_SVM)),
        np.sqrt(mean_squared_error(y_test, y_pred_KNN)),
        np.sqrt(mean_squared_error(y_test, y_pred_GD)),
        np.sqrt(mean_squared_error(y_test, y_pred_RF))
    ],

    'R2 Score': [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_DT),
        r2_score(y_test, y_pred_SVM),
        r2_score(y_test, y_pred_KNN),
        r2_score(y_test, y_pred_GD),
        r2_score(y_test, y_pred_RF)
    ]
})

# Sort by R2 Score
results = results.sort_values(
    by='R2 Score',
    ascending=False
)

# Reset index
results = results.reset_index(drop=True)

print("\nMODEL COMPARISON")
print(results)

# Save final model
folder = r"D:\python\DATA_SCIENCE\MACHINE_LEARNING\SuperVisedLearning\carpriceproject"

# joblib.dump(model_lr, "car_price_model.pkl")
# joblib.dump(scalar,"scaler.pkl")
# joblib.dump(X.columns.tolist(),"columns.pkl")

print("All files saved successfully!")


# https://car-price-prediction-kntkyb7cyxjqsgnrkebbos.streamlit.app/