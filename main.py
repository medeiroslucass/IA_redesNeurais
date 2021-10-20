import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder


csv = pd.read_csv('dados.csv', sep=",")

#Label encoder
le = LabelEncoder()
csv["fruta"] = le.fit_transform(csv["fruta"])

#Dados de entrada e saida
dados = csv.values
atributos = dados[:,2:]
saida = dados[:,1]


#Modelo
modelo = Sequential()
modelo.add(Dense(units=2, activation="linear")) #Camada Oculta (processamento)
modelo.add(Dense(units=1, activation='sigmoid')) #Camada de Saida (respossta)


# #compilador
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# #Treinar
modelo.fit(atributos,saida,batch_size=1667 , epochs=500)


#Predict
vetor = [
    [3.1, 122],
    [4.1, 146],
    [2.2, 86],
]

resultado = modelo.predict(vetor)




for _ in resultado:
    if _ < 0.5:
        print("Laranja")
    elif _ >= 0.5:
        print("Limao")