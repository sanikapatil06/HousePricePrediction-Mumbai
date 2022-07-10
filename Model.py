import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import pickle as pk
from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot as plt
import seaborn as sb

rfc = RandomForestRegressor()

le = LabelEncoder()
df['Location'] = le.fit_transform(df['Location'])
df['Price'] = np.log(df['Price'])
x = df.drop(["id","Price", "Lift Available", "Maintenance Staff",
             "24x7 Security", "Children's Play Area", "Intercom",
             "Indoor Games", "Landscaped Gardens", "Clubhouse"
             ], axis =1)
y = df["Price"]

q1 = x['Area'].quantile(0.25)
q3 = x['Area'].quantile(0.75)

iqr = q3-q1

u = q3 + 1.5*iqr
l = q1 - 1.5*iqr

out1 = x[x['Area'] < l].values
out2 = x[x['Area'] > u].values

x['Area'].replace(out1,l,inplace = True)
x['Area'].replace(out2,u,inplace = True)

q1_1 = df['Price'].quantile(0.25)
q3_1 = df['Price'].quantile(0.75)

iqr_1 = q3_1-q1_1

u_1 = q3_1 + 1.5*iqr_1
l_1 = q1_1 - 1.5*iqr_1

out1_1 = df[df['Price'] < l_1].values
out2_1 = df[df['Price'] > u_1].values

df['Price'].replace(out1_1,l_1,inplace = True)
df['Price'].replace(out2_1,u_1,inplace = True)

sb.boxplot(x['Area'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0,
                                                    test_size=0.3)

rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

print("Score:", r2_score(y_test, y_pred))

file = "Trained_model.pkl"
pk.dump(rfc, open(file, "wb"))

