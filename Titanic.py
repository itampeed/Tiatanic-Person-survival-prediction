import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('tested.csv')

#Handling the missing values
data.drop('Cabin', axis=1, inplace = True)
data['Age'].fillna(data['Age'].median(), inplace = True)
data['Fare'].fillna(data['Fare'].median(), inplace = True)

df = {'male':0, 'female':1}
data['Sex'] = data['Sex'].map(df)

#selecting the features
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = data['Survived']

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#training the model
model = LinearRegression()
model.fit(X_train, y_train)

predicted = model.predict([[2,1,25,0,1,70]])
print(predicted)
