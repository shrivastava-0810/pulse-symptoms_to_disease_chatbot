import numpy as np
import pandas as pd
from collections import OrderedDict as od

description = pd.read_csv('D:/ML_projects/symptom_description.csv')
precaution = pd.read_csv('D:/ML_projects/symptom_precaution.csv')

diseases = description.iloc[:, 0].values
desc = description.iloc[:, 1:].values

prec = precaution.iloc[:, 1:].values

from clusters import other_possible_symptoms
from classification_algo import calc_prob

main_symptoms_given = []
print('Hey there, this is Dr. Pulse, your virtual doctor')
symptoms_input = list(input('Kindly enter symptoms (atleast two): ').split())
main_symptoms_given.extend(symptoms_input)

possible_symptoms = other_possible_symptoms(symptoms_input)

while possible_symptoms:
    symptoms_input_total = []
    j = 0
    while len(possible_symptoms)>20:
        i = 5
        print('\nSelect the symptoms that you experiencing:')
        print(possible_symptoms[:i])
        symptoms_input = list(input().split())
        if symptoms_input[0] in possible_symptoms[:i]:
            symptoms_input_total.extend(symptoms_input)
        possible_symptoms = possible_symptoms[i:]
        j+=1
    main_symptoms_given.extend(symptoms_input_total)
    if j>=3:
        possible_symptoms = list(set(possible_symptoms) & set(other_possible_symptoms(symptoms_input_total)))
        
    if possible_symptoms:
        print('\nAre suffering from ' + possible_symptoms[0])
        if input() == 'yes':
            main_symptoms_given.append(possible_symptoms[0])
            possible_symptoms = list(set(possible_symptoms) & set(other_possible_symptoms([possible_symptoms.pop(0)])))
        else:
            possible_symptoms.pop(0)

predicted_diseases, probabilities = calc_prob(main_symptoms_given)

chances = list(probabilities)

predicted_data = od()

for i in range(len(probabilities)):
    if probabilities[i] >= 0.75:
        chances[i] = 'Very high chances'
    elif 0.5 <= probabilities[i] < 0.75:
        chances[i] = 'High chances'
    elif 0.25 <= probabilities[i] < 0.5:
        chances[i] = 'Moderate chances'
    else:
        chances[i] = 'Low chances'

for i in range(len(predicted_diseases)):
    data = []
    index = list(diseases).index(predicted_diseases[i].strip())
    data.append(chances[i])
    data.append(desc[index][0])
    data.append(list(prec[index]))
    predicted_data[predicted_diseases[i]] = data

print('\nDiseases you may be suffering from: ')
print(predicted_data)
print('\nSymptoms provided:')
print(main_symptoms_given)
