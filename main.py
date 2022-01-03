#######################################
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
Y = music_data['genre']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

score = accuracy_score(Y_test, predictions)
score

############################################
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib as jb

# music_data = pd.read_csv('music.csv')
# X = music_data.drop(columns=['genre'])
# Y = music_data['genre']

# model = DecisionTreeClassifier()
# model.fit(X, Y)

model = jb.load('music-recommender.joblib')
predictions = model.predict( [ [21, 1] ] )
predictions

############################################
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
Y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, Y)

tree.export_graphviz(model, out_file='music-recommender.dot',
                    feature_names=['age', 'gender'],
                    class_names=sorted(Y.unique()),
                    label='all',
                    rounded=True,
                    filled=True)