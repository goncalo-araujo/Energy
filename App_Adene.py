#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required packages from python library
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


X = pd.read_pickle("./X_train.pkl")


# In[3]:


y = pd.read_pickle("./y_train.pkl")


# In[4]:


#Here we import the ExtraTreesRegressor model from sklearn package, nad train it to fit pickled data.
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor(n_jobs=-1, random_state=42)
model.fit(X,y)


# In[5]:


#Here we define the functions that translate sidebar inputs to numerical values for the machine learning model.
#In this case, we translate year of construction to construction archetype.

def period_to_numeric(a):
    if a < 1919:
        return 0
    if a > 1918 and a < 1946:
        return 1
    if a > 1945 and a < 1961:
        return 2
    if a > 1960 and a < 1971:
        return 3
    if a > 1970 and a < 1981:
        return 4
    if a > 1980 and a < 1991:
        return 5
    if a > 1990 and a < 1996:
        return 6
    if a > 1995 and a < 2001:
        return 7
    if a > 2000 and a < 2006:
        return 8
    else:
        return 9


# In[6]:


#Here we translatic domestic hot water values from energy source to numerical.
def dhw_to_numeric(a):
    if a == 'Gás Natural':
        return 1
    if a == 'Electricidade':
        return 2
    if a == 'Urban network - Climaespaço':
        return 3
    if a == 'Solar':
        return 4
    if a == 'Gás Propano':
        return 5
    if a == 'Diesel':
        return 6
    if a == 'Biomassa':
        return 7
    if a == 'Gás Butano':
        return 8
    else:
        return 9 #Biomassa sólida


# In[7]:


#Here we translate respective wall u values (kWh/m2) for each construction archetype
def period_to_wall_u(a):
    if a < 1919:
        return 2.90
    if a > 1918 and a < 1946:
        return 2.90
    if a > 1945 and a < 1961:
        return 1.20
    if a > 1960 and a < 1971:
        return 0.96
    if a > 1970 and a < 1981:
        return 1.10
    if a > 1980 and a < 1991:
        return 0.47
    if a > 1990 and a < 1996:
        return 0.50
    if a > 1995 and a < 2001:
        return 0.59
    if a > 2000 and a < 2006:
        return 0.45
    else:
        return 0.179


# In[8]:


st.write("""
# Consumo energético em Lisboa

Esta app prevê o consumo energético da sua casa/edifício em Lisboa
""")
st.write("---")


# In[9]:


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Especifique as caracteristicas do seu apartamento ou edificio')

def user_input_features():
    B_type = 0
    Period = st.sidebar.slider('Ano de Construção', 1918, 2021, 1960)
    f_area = st.sidebar.slider('Área útil', round(50, 2), round(400, 2), round(100, 2))
    f_height = st.sidebar.slider('Pé direito', 2.00, 4.00, 2.75)
    Typology = st.sidebar.slider('Assoalhadas', 0, 7, 3)
    N_floors = st.sidebar.slider('Número de pisos', 1, 15, 6)
    Window_area = st.sidebar.slider('Área janelas', 0.00, 100.00, 25.00)
    Window_average_U_value = st.sidebar.slider('U value janela', 0.50, 4.00, 3.20)
    DHW = st.sidebar.select_slider('Águas quentes e sanitárias', ["Electricidade", "Urban network - Climaespaço", "Solar", "Gás Natural", "Gás Propano", "Gás Butano", "Diesel", "Biomassa", "Biomassa sólida"], "Electricidade")
    gT = st.sidebar.slider('Transmissividade do vidro', 0.00, 1.00, 0.31)
    data = {'Tipologia': 0,
             'Ano de Construção': period_to_numeric(Period),
             'Área útil': f_area,
             'Pé direito': float(f_height),
             'Assoalhadas': Typology,
             'Número de pisos': N_floors,
             'Área paredes': np.sqrt(f_area)*f_height,
             'U value parede': period_to_wall_u(Period),
             'Área janelas': Window_area,
             'Rácio de envidraçado': Window_area/(np.sqrt(f_area)*f_height),
             'U value janela': Window_average_U_value,
             'Águas quentes e sanitárias': dhw_to_numeric(DHW),
             'Transmissividade do vidro': gT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


# In[10]:


#Define prediction function that will be applied to user selected inputs
def prediction(dataframe):  
    prediction = model.predict(dataframe)
    print(prediction)
    return prediction


# In[11]:


#df


# In[12]:


#prediction(df)


# In[13]:


st.header('Previsão do consumo energético total anual')
st.write(round(prediction(df)[0], 2), "kWh/m2")
st.write(round(prediction(df)[0]*df["Área útil"][0], 2), "kWh")
st.write('---')


# In[27]:


st.header('Especifique o/os tipo/os de renovação')
#up = st.selectbox("Tipo de renovação", ["Paredes", "Janelas", "Fonte da sua energia"])
wall_checkbox = st.checkbox("Paredes")
window_checkbox = st.checkbox("Janelas")
energy_checkbox = st.checkbox("Águas quentes e sanitárias")

checkboxes = [wall_checkbox, window_checkbox, energy_checkbox]


# In[28]:


def upgrade_wall_features(checkbox):
        wt = []
        if checkbox == True:
            wt = st.selectbox("Tipo de solução Construtiva", ["perfil em aço galvanizado 15 cm, isolamento com 2 painéis (2x6) cm, painel OSB 1.1 cm,  EPS 4 cm",
                                                              "ETIC: tijolo de 22 cm, poliestireno moldado expandido de 6 cm, revestimento em reboco armado de 2 cm",
                                                              "Fachada ventilada: tijolo de 22 cm,  isolamento térmico (4 cm XPS), e caixa de ar ventilada de 3 cm",
                                                              "Alvenaria de tijolo (11+11) cm com caixa de ar de 7 cm parcialmente preenchida (4 cm XPS)",
                                                              "Alvenaria de tijolo (15+11) cm com caixa de ar de 7 cm parcialmente preenchida (4 cm XPS)"])
            return wt
        else:
            return wt

def upgrade_windowu_features(checkbox):
        wu = []
        if checkbox == True:
            wu = st.slider('Novo U value da sua janela', 0.50, 4.00, 3.20)
            return wu
        else:
            return wu

        
def upgrade_windowgt_features(checkbox):
        ngt = []
        if checkbox == True:
            ngt = st.slider("Transmissividade do novo vidro", 0.00, 1.00, 0.31)
            return ngt
        else:
            return ngt
        
def upgrade_energy_features(checkbox):   
        nes = []
        if checkbox == True:
            nes = st.select_slider('Águas quentes e sanitárias', ["Electricidade", "Urban network - Climaespaço", "Solar", "Gás Natural", "Gás Propano", "Gás Butano", "Diesel", "Biomassa", "Biomassa sólida"], "Gás Natural")
            return nes
        else:
            return nes

wf = upgrade_wall_features(wall_checkbox)
wiuf = upgrade_windowu_features(window_checkbox) 
wigf = upgrade_windowgt_features(window_checkbox)
ef = upgrade_energy_features(energy_checkbox)


# In[29]:


def upgrade_wall_uvalue(selection):
    if selection == "perfil em aço galvanizado 15 cm, isolamento com 2 painéis (2x6) cm, painel OSB 1.1 cm,  EPS 4 cm":
        return 0.179
    if selection == "ETIC: tijolo de 22 cm, poliestireno moldado expandido de 6 cm, revestimento em reboco armado de 2 cm":
        return 0.45
    if selection == "Fachada ventilada: tijolo de 22 cm,  isolamento térmico (4 cm XPS), e caixa de ar ventilada de 3 cm":
        return 0.59 
    if selection == "Alvenaria de tijolo (11+11) cm com caixa de ar de 7 cm parcialmente preenchida (4 cm XPS)":
        return 0.50 
    else:
        return 0.47 


# In[30]:


def upgrade_wall_df(checkbox):
    
        if checkbox == True:
            return upgrade_wall_uvalue(wf)
        else:
            return df["U value parede"]


def upgrade_windowu_df(checkbox):
        if checkbox == True:
            return wiuf
        else:
            return df["U value janela"]

def upgrade_windowgt_df(checkbox):
        if checkbox == True:
            return wigf
        else:
            return df["Transmissividade do vidro"]
        
def upgrade_energy_df(checkbox):
        if checkbox == True:
            return dhw_to_numeric(ef)
        else:
            return df["Águas quentes e sanitárias"]


# In[31]:


newdf = df.copy()


# In[32]:


newdf["U value parede"] = upgrade_wall_df(wall_checkbox)


# In[33]:


newdf["U value janela"] = upgrade_windowu_df(window_checkbox)


# In[34]:


newdf["Transmissividade do vidro"] = upgrade_windowgt_df(window_checkbox)


# In[35]:


newdf["Águas quentes e sanitárias"] = upgrade_energy_df(energy_checkbox)


# In[40]:


#newdf


# In[36]:


st.write('---')


# In[54]:


preds = pd.DataFrame([round(prediction(df)[0]*df["Área útil"][0], 2), round(prediction(newdf)[0]*df["Área útil"][0], 2)])


# In[57]:


preds.set_axis(["Actual", "Após renovação"], axis=0, inplace=True)
preds.set_axis(["kWh"], axis=1, inplace=True)


# In[62]:


preds["kWh"][1]


# In[49]:


st.bar_chart(preds)


# In[65]:


savings = round((preds["kWh"][0]-preds["kWh"][1])*0.15, 2)
st.write("**Você poupa **", savings, "**euros assumindo 1 kWh = 0.15 €**")


# In[ ]:


# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)


# In[ ]:


# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')


# In[ ]:


# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')


# In[ ]:





# In[ ]:




