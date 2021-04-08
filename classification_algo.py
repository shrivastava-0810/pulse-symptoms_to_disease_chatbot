import pandas as pd
import numpy as np

dataset = pd.read_csv('D:/ML_projects/Training.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y.reshape(-1, 1)).toarray()

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 132))

classifier.add(Dense(units = 41, kernel_initializer = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X, y, batch_size = 64, epochs = 5)

classifier.save('D:/ML_projects/symptoms_classifier')

# y_pred = (y_pred > 0.5)

#Naive bayes 
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X, y)

#Logistic Regression
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X, y)

#Random Forest
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 50, criterion = "entropy", 
#                                     random_state = 0)
# classifier.fit(X, y)

# y_pred = classifier.predict(X_test)
# y_pred = label_encoder.inverse_transform(y_pred)

def calc_prob(symptoms):
    X_test = np.zeros(X.shape[1])
    for symptom in symptoms:
        index = dataset.columns.get_loc(symptom)
        X_test[index] = 1
        
    X_test = X_test.reshape(1,-1)
    classifier = keras.models.load_model('D:/ML_projects/symptoms_classifier')
    y_pred = classifier.predict_proba(X_test)
    top_five_indexes = y_pred.argsort()[0, :][-5:][::-1]
    top_five = label_encoder.inverse_transform(top_five_indexes)
    probability = y_pred[0, :][top_five_indexes]
    return(top_five, probability)
