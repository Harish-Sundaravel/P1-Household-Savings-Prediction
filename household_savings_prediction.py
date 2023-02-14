import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/content/family_expenses.csv',sep='\t')

df.head()

df.tail()

df.info()

df.describe()

savings=df.savings
#savings
savings.plot()
plt.show()

correlation=df.corr()

plt.figure(figsize=(8,8))
sns.heatmap(correlation,cbar=True,square=True,annot=True,annot_kws={'size':8})

print(correlation['savings'])

sns.distplot(df['savings'])

from sklearn.model_selection import train_test_split

x=df.drop(['month_year','savings'],axis=1)
y=df.savings
print(x)
print('-----------------')
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor(n_estimators=100)

regressor.fit(x_train,y_train)

test_data_prediction=regressor.predict(x_test)
print(test_data_prediction)

x_test

q=x_test.index
p=df.loc[q,'savings']
df1=pd.DataFrame(list(zip(p,test_data_prediction)),
                      columns=['Actual','Predicted'],
                 index=q)
df1

df2=pd.DataFrame(list(zip(y_test,test_data_prediction)),columns=['Actual','Predicted'],index=q)
df2

from sklearn import metrics

error=metrics.r2_score(y_test,test_data_prediction)
print(error)

y_test=list(y_test)
plt.plot(y_test,color='blue',label='original ')
plt.plot(test_data_prediction,color='green',label='predicted')
plt.legend()
plt.xlabel('features')
plt.ylabel('ssavings ')
plt.title('actual vs predicted values')
plt.show()

