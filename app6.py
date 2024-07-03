#!/usr/bin/env python
# coding: utf-8

# In[139]:


import streamlit as st
from datetime import datetime

def test(datum,berichttype,soortbericht):
    import pandas as pd
    import numpy as np
    import joblib
    model = joblib.load(r"C:\Users\masfo\Get Fit Girl\model.pkl")
    # Convert data into DataFrame
    soortbericht=soortbericht
    berichttype=berichttype
    df_tijd=pd.DataFrame([{'datum':datum,'berichttype':berichttype, 'soortbericht':soortbericht}])
    # Convert 'datum' column to datetime
    df_tijd['datum'] = pd.to_datetime(df_tijd['datum'], dayfirst=True)
    # Extract the desired information
    df_tijd['maand'] = df_tijd['datum'].dt.month
    df_tijd['dag_van_de_week'] = df_tijd['datum'].dt.dayofweek
    df_tijd['dag_van_de_maand'] = df_tijd['datum'].dt.day
    df_tijd=df_tijd.drop(columns='datum')
    minuut = np.arange(0, 60)
    tijd = np.arange(7, 20)
    maand=df_tijd['maand'].values[0]
    dag_van_de_week=df_tijd['dag_van_de_week'].values[0]
    dag_van_de_maand=df_tijd['dag_van_de_maand'].values[0]
    # Create a DataFrame with all combinations of hour and minute
    possible_times = pd.DataFrame([(m, u, dag_van_de_week, dag_van_de_maand, maand) 
                               for m in minuut
                               for u in tijd], 
                              columns=['minuut', 'tijd', 'dag_van_de_week','dag_van_de_maand','maand'])
    possible_times[berichttype] = 1  # Example for Instagram reel
    possible_times[soortbericht] = 1  # Example for 'Recept hartig'


    # Define all content type columns and set them to 0 by default
    dummy_columns = ['Berichttype_IG-reel','Berichttype_Instagram-afbeelding', 'Berichttype_Instagram-carrousel', 'Soort Bericht_Black Friday', 'Soort Bericht_Boek',
                 'Soort Bericht_Bruiloft Fred', 'Soort Bericht_Giveaway', 'Soort Bericht_Grappig',
                 'Soort Bericht_Info GFG', 'Soort Bericht_Informatie producten', 'Soort Bericht_Motivatie',
                 'Soort Bericht_Motivatie Fred', 'Soort Bericht_Motivatie Grappig', 'Soort Bericht_Motivatie Wen',
                 'Soort Bericht_Motivatie Wen&Fred', 'Soort Bericht_Overzicht recepten', 'Soort Bericht_Recept Zoet',
                 'Soort Bericht_Recept hartig', 'Soort Bericht_What I eat in a day', 'Soort Bericht_Samenwerking', 'Soort Bericht_Team', 'Soort Bericht_Tips',
                 'Soort Bericht_Afvalfoto lid', 'Soort Bericht_Zwanger Fred', 'Soort Bericht_Zwanger Wen',
                 'Soort Bericht_beurs', 'Soort Bericht_challenge', 'Soort Bericht_kinderen']
    dummy_columns.remove(berichttype)
    dummy_columns.remove(soortbericht)
    for col in dummy_columns:
        possible_times[col] = 0

    column_order=['dag_van_de_week', 'dag_van_de_maand', 'maand', 'tijd', 'minuut', 'Berichttype_IG-reel',
       'Berichttype_Instagram-afbeelding', 'Berichttype_Instagram-carrousel',
       'Soort Bericht_Afvalfoto lid', 'Soort Bericht_Black Friday',
       'Soort Bericht_Boek', 'Soort Bericht_Bruiloft Fred',
       'Soort Bericht_Giveaway', 'Soort Bericht_Grappig',
       'Soort Bericht_Info GFG', 'Soort Bericht_Informatie producten',
       'Soort Bericht_Motivatie', 'Soort Bericht_Motivatie Fred',
       'Soort Bericht_Motivatie Grappig', 'Soort Bericht_Motivatie Wen',
       'Soort Bericht_Motivatie Wen&Fred', 'Soort Bericht_Overzicht recepten',
       'Soort Bericht_Recept Zoet', 'Soort Bericht_Recept hartig',
       'Soort Bericht_Samenwerking', 'Soort Bericht_Team',
       'Soort Bericht_Tips', 'Soort Bericht_What I eat in a day',
       'Soort Bericht_Zwanger Fred', 'Soort Bericht_Zwanger Wen',
       'Soort Bericht_beurs', 'Soort Bericht_challenge',
       'Soort Bericht_kinderen']
    possible_times=possible_times[column_order]
    y_pred=model.predict(possible_times)
    possible_times['Succes_Score_Prediction'] = y_pred

# Find the maximum predicted success score
    max_score = possible_times['Succes_Score_Prediction'].max()

# Filter the rows with the maximum score
    max_score_times = possible_times[possible_times['Succes_Score_Prediction'] == max_score]

# Get unique values in 'tijd' and 'minuut' columns
    unique_tijden = max_score_times['tijd'].unique()
    unique_minuten = max_score_times['minuut'].unique()
    return unique_tijden, unique_minuten

    

def main():
    st.title('Beste tijd om te posten')

    datum = st.date_input('Select a Date', datetime.now())
    berichttype = st.selectbox('Select Berichttype:', ['Berichttype_IG-reel', 'Berichttype_Instagram-afbeelding','Berichttype_Instagram-carrousel'])
    soortbericht= st.selectbox('Select Berichtsoort:',[ 'Soort Bericht_Recept Zoet', 'Soort Bericht_Recept hartig', 'Soort Bericht_Samenwerking', 'Soort Bericht_Team','Soort Bericht_Tips', 'Soort Bericht_What I eat in a day'])

    if st.button('Go'):
        resultaat1, resultaat2=test(datum,berichttype,soortbericht)
        st.write(f"Uren met de hoogste score:{resultaat1} en Minuten met de hoogste score:{resultaat2}")

if __name__ == '__main__':
    main()


# In[ ]:




