import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Приложение для предсказания вида пингвина!

Предсказвает вид **пингвинов Палмера**.

Данные взяты из [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) (автор - Allison Horst).
""")

img = Image.open('Palmer_penguins.png')
st.image(img, width=400)

st.sidebar.header('Выбор характеристик:')

st.sidebar.markdown("""
[Пример CSV-файла](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Собираем датафрейм
uploaded_file = st.sidebar.file_uploader("Загрузить CSV-файл", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Остров обитания',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Пол',('male','female'))
        bill_length_mm = st.sidebar.slider('Длина тела (мм)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Толщина тела (мм)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Длина плавника (мм)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Масса тела (г)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Соединяем наши фичи имеющимся датасетом
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('Выбранные характеристики пингвина:')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Подождите, пока CSV-файл загрузится. Текущие выбранные параметры представлены ниже.')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Предсказание вида пингвина:')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Вероятность предсказания по каждому виду:')
st.write(prediction_proba)
