import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import streamlit as st

@st.experimental_memo(suppress_st_warning=True)
def load_data():
    '''
    Load data from the Data folder 
    '''
    df = pd.read_csv('Data/players_20.csv')
    return df

def data_manu(df):
    '''
    Data manipulation 
    -------------------
    Input   :   df-> DataFrame
    Return  :   df-> The cleaned data as DataFrame
                sem_df-> The semi cleaned Data for plotting purpose
    '''
    # First remove commas from the column 
    df['player_positions']=df['player_positions'].str.replace(',','')
    df['player_positions'] = df['player_positions'].astype(str).str.split().str[0]
    #
    att = dict.fromkeys(['ST', 'LW', 'RW', 'LS', 'RS', 'CF', 'RF', 'LF'], 'Attacker')
    mid = dict.fromkeys(['CM', 'RM', 'LM', 'CAM', 'CDM', 'LCM', 'RCM', 'RDM', 'LDM', 'RAM', 'LAM'], 'Midfielder')
    dfnc = dict.fromkeys(['CB', 'LB', 'RB', 'RCB', 'LCB', 'RWB', 'LWB' ], 'Defender')

    df.player_positions.replace('GK', 'Goalkeeper', inplace=True)
    df.player_positions.replace(att, inplace=True)
    df.player_positions.replace(mid, inplace=True)
    df.player_positions.replace(dfnc, inplace=True)  
    sem_df = df  
    # remove goal keeper from dataset
    df = df[df['player_positions'] != 'Goalkeeper']
    # Combining Midfielder and Attacker into one
    df['position'] = df['player_positions'].apply(lambda x: 'defense' if 'Defender' in x else 'midfield') 
    return df, sem_df

def run_model(df):
    '''
    Running the model
    '''
    df = df[['position', 'shooting','passing','dribbling','defending']]
    X  = df.drop('position', axis=1).values
    y  = df['position'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn, X_test, y_test

## ----- App ----- ##
st.set_page_config(page_title="Technble-Demo-App", layout="wide")
st.title("Preditcs player's playing position")
# Loading Data
data = st.sidebar.checkbox(label = 'Load Data', value = False)  

if data :
    df = load_data()
    manu_df, sem_df = data_manu(df)
    with st.expander(label = 'Data', expanded=False):
        st.info("Data reference - Data/players_20")
        st.dataframe(df)
    with st.expander(label = 'Visualization', expanded=True):     
        col1 , _ ,col2 = st.columns([1,0.5,1])
    with col1:
        fig, ax = plt.subplots(figsize=(5,5))
        ax = sem_df.player_positions.value_counts().plot(kind = 'pie',autopct = '%0.1f%%',shadow = True, cmap = 'Set3')
        plt.title('Position Representation\n', fontsize = 16 )
        plt.xlabel('')
        plt.ylabel('')
        plt.axis('equal')
        plt.show() 
        st.pyplot(fig) 
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5,5))
        ax2 = manu_df.position.value_counts().plot(kind = 'pie',autopct = '%0.1f%%',shadow = True, cmap = 'Set3')
        plt.title('Combined Midfiled+Attack ', fontsize = 14 )
        plt.xlabel('')
        plt.ylabel('')
        plt.axis('equal')
        plt.show() 
        st.pyplot(fig2)
    
    player = st.sidebar.selectbox(label="Player", options=manu_df['short_name'])
    sel = df = df[df['short_name'].str.contains(player)]
    attr = sel[[ 'shooting','passing','dribbling','defending']]

    with st.expander(label = 'Model', expanded=True):
        st.info('Selected Player: ' + player)
        st.write("Selected player's attributes:")
        st.dataframe(pd.DataFrame(attr.to_numpy(), columns=[ 'shooting','passing','dribbling','defending'],index=[player]))  
        run = st.button(label='Run')
        if run:
            with st.spinner('Running K-Nearest Neighbor Algorithim ...'):
                time.sleep(0.5)
                res , x, y = run_model(manu_df)
            st.success('Accuracy of your model: ' + str(round(res.score(x,y),3)*100) + '%')
            st.info(player+' is playing in : '+ str(res.predict(attr)))
