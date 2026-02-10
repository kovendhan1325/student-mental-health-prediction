
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = {
    'Age':[21,22,20,23,21,24],
    'StudyHours':[5,8,2,6,7,3],
    'SleepHours':[7,4,8,6,5,4],
    'AcademicPerformance':[75,60,85,70,65,50],
    'Alcoholic':['No','Yes','No','No','Yes','Yes'],
    'Risk':['Low','High','Low','Low','Medium','High']
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['Alcoholic'] = le.fit_transform(df['Alcoholic'])
df['Risk'] = le.fit_transform(df['Risk'])

X = df.drop('Risk',axis=1)
y = df['Risk']

model = RandomForestClassifier()
model.fit(X,y)

joblib.dump(model,"model/model.pkl")
print("Model trained & saved")
