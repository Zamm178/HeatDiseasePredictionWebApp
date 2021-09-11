import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('./heart.csv')
x = data.drop(columns='target', axis=1)
y = data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

model=RandomForestClassifier(n_estimators=20)
model.fit(x_train, y_train)
y_pred= model.predict(x_test)
p = model.score(x_test,y_test)
print(p)

file = 'model.pkl'
pickle.dump(model, open(file, 'wb'))
