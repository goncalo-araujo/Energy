#!/usr/bin/env python
# coding: utf-8

# In[56]:


#Import required packages from python library
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'


# In[57]:


X = pd.read_pickle("./X_train.pkl").reset_index().drop("index", axis=1)#.drop("B_type", axis=1)


# In[58]:


#X


# In[59]:


y = pd.read_pickle("./y_train.pkl")


# In[60]:


# X["f_area"] = X["f_area"].apply(log)
# X["f_height"] = X["f_height"].apply(log)
# X["Wall_area"] = X["Wall_area"].apply(log)
# X["Wall_average_U-value"] = X["Wall_average_U-value"].apply(log)
# X["Window_area"] = X["Window_area"].apply(log)
# X["wwr"] = X["wwr"].apply(log)
# X["Window_average_U-value"] = X["Window_average_U-value"].apply(log)
# X["gtvi"] = X["gtvi"].apply(log)
# X["gT"] = X["gT"].apply(log)


# In[61]:


from sklearn.model_selection import train_test_split,_validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[62]:


#Here we import the ExtraTreesRegressor model from sklearn package, nad train it to fit pickled data.
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
model = ExtraTreesRegressor(n_jobs=-1, random_state=42)
model.fit(X_train,y_train)


# In[63]:


from sklearn.metrics import r2_score,mean_squared_error


# In[64]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = np.sqrt(mean_squared_error(test_labels, predictions))
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} kWh/m2.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('R2 Score = {:0.2f}'.format(r2_score(test_labels, predictions)))


# In[65]:


st.write("Model Performance")
st.write(" Average Error: 27.9629 kWh/m2.")
st.write("Accuracy = 73.62%")
st.write("R2 Score = 0.71")


# In[66]:


st.write("""
# Consumo energético de edificios em Lisboa

Esta app prevê o consumo energético de multiplos edificios em Lisboa, de acordo com os seus parametros
""")
st.write("---")


# In[67]:


# Sidebar
# Header of Specify Input Parameters


# In[68]:


spectra = st.file_uploader("upload file", type={"csv", "txt"})
if spectra is not None:
    df = pd.read_csv(spectra)
else:
    df = X.sample(20).reset_index().drop("index", axis=1)
st.write(df)


# In[69]:


# wt = ["perfil em aço galvanizado 15 cm, isolamento com 2 painéis (2x6) cm, painel OSB 1.1 cm,  EPS 4 cm",
#                   "ETIC: tijolo de 22 cm, poliestireno moldado expandido de 6 cm, revestimento em reboco armado de 2 cm",
#                   "Fachada ventilada: tijolo de 22 cm,  isolamento térmico (4 cm XPS), e caixa de ar ventilada de 3 cm",
#                   "Alvenaria de tijolo (11+11) cm com caixa de ar de 7 cm parcialmente preenchida (4 cm XPS)",
#                   "Alvenaria de tijolo (15+11) cm com caixa de ar de 7 cm parcialmente preenchida (4 cm XPS)"]


# In[70]:


# st.header('Especifique o/os tipo/os de renovação para optimização')
# #up = st.selectbox("Tipo de renovação", ["Paredes", "Janelas", "Fonte da sua energia"])
# wall_checkbox = st.checkbox("Paredes", True)
# window_checkbox = st.checkbox("Janelas", True)

# st.write("---")


# In[71]:


st.write("Introduza o U-value da(s) nova(s) parede(s):")
wf = st.text_input("Wall U-Value", "0.179, 0.45, 0.50")


st.write("Introduza o preço/m2 respectivo da(s) nova(s) parede(s):")
pwf = st.text_input("Respective prices", "60, 30, 26")


st.write("Introduza o U-value da(s) nova(s) janela(s):")
wif = st.text_input("U-value da(s) janela(s)", "1.98, 0.95")


st.write("Introduza a transmissividade do(s) novo(s) vidro(s):")
wif2 = st.text_input("transmissividade(gT) do(s) vidro(s)", "0.70, 0.31")



st.write("Introduza o preço/m2 respectivo da(s) nova(s) caixilharia(s):")
pwif = st.text_input("Preços respectivos", "75, 100")


# In[72]:


wall_inputs = wf.split(",")
floats_walls_u = [0]
for i in enumerate(wall_inputs):
    floats_walls_u = np.append(floats_walls_u, float(wall_inputs[i[0]]))
#floats_walls_u


# In[73]:


wall_prices = pwf.split(",")
floats_walls_prices = [0]
for i in enumerate(wall_prices):
    floats_walls_prices = np.append(floats_walls_prices, float(wall_prices[i[0]]))
#floats_walls_prices


# In[74]:


window_inputs_u = wif.split(",")
floats_windows_u = [0]
for i in enumerate(window_inputs_u):
    floats_windows_u = np.append(floats_windows_u, float(window_inputs_u[i[0]]))
#floats_windows_u


# In[75]:


window_inputs_gt = wif2.split(",")
floats_windows_gt = [0]
for i in enumerate(window_inputs_gt):
    floats_windows_gt = np.append(floats_windows_gt, float(window_inputs_gt[i[0]]))
#floats_windows_gt


# In[76]:


window_prices = pwif.split(",")
floats_windows_prices = [0]
for i in enumerate(window_prices):
    floats_windows_prices = np.append(floats_windows_prices, float(window_prices[i[0]]))
#floats_windows_prices


# In[77]:


ncomb = len(floats_windows_u)*len(floats_walls_u)
#ncomb


# In[78]:


floats_windows= []
for i in enumerate(floats_windows_u):
    floats_windows = np.append(floats_windows, [floats_windows_u[i[0]], floats_windows_gt[i[0]], floats_windows_prices[i[0]]])


# In[79]:


floats_windows = floats_windows.reshape(len(floats_windows_u), 3)
#floats_windows


# In[80]:


floats_walls = []
for i in enumerate(floats_walls_u):
    floats_walls = np.append(floats_walls, [floats_walls_u[i[0]], floats_walls_prices[i[0]]])


# In[81]:


floats_walls = floats_walls.reshape(len(floats_walls_u), 2)
#floats_walls


# In[82]:


arr = [[e1, e2] for e1 in floats_walls for e2 in floats_windows]


# In[91]:


#arr


# In[89]:


def opt_df(x):
    newdf = df.copy()
    newdf["cost"] = np.repeat(0, len(x))
    for i in enumerate(x):
        newdf["Wall_average_U_value"].iloc[i[0]] = arr[i[1]][0][0]
        newdf["Window_average_U_value"].iloc[i[0]] = arr[i[1]][1][0]
        newdf["gT"].iloc[i[0]] = arr[i[1]][1][1]
        newdf["cost"].iloc[i[0]] = arr[i[1]][1][2]*newdf["Window_area"].iloc[i[0]] + arr[i[1]][0][1]*newdf["Wall_area"].iloc[i[0]]
        if newdf["Wall_average_U_value"].iloc[i[0]] == 0 or newdf["Wall_average_U_value"].iloc[i[0]] > df["Wall_average_U_value"].iloc[i[0]]:
            newdf["Wall_average_U_value"].iloc[i[0]] = df["Wall_average_U_value"].iloc[i[0]]
            
        if newdf["Window_average_U_value"].iloc[i[0]] == 0 or newdf["Window_average_U_value"].iloc[i[0]] > df["Window_average_U_value"].iloc[i[0]]:
            newdf["Window_average_U_value"].iloc[i[0]] = df["Window_average_U_value"].iloc[i[0]]
            newdf["gT"].iloc[i[0]] = df["gT"].iloc[i[0]]
    mean = np.mean(model.predict(newdf.drop("cost", axis=1)))
    #predictions = model.predict(newdf.drop("cost", axis=1))
    std = np.std(model.predict(newdf.drop("cost", axis=1)))
    costs = np.sum(newdf["cost"])
    return [mean, std, costs]


# In[ ]:





# In[ ]:





# In[93]:


st.write("---")
st.write("original average energy consumption, standard deviation, and total retrofit cost:")
st.write(str( round(opt_df(np.repeat(0, 20))[0], 2)) + " kWh/m2, " + str( round(opt_df(np.repeat(0, 20))[1], 2)) + " kWh/m2, " + str( round(opt_df(np.repeat(0, 20))[2], 2)) + " €")
st.write("full retrofit energy consumption, standard deviation, and cost:")
st.write(str( round(opt_df(np.repeat(5, 20))[0], 2)) + " kWh/m2, " + str( round(opt_df(np.repeat(5, 20))[1], 2)) + " kWh/m2, " + str( round(opt_df(np.repeat(5, 20))[2], 2)) + " €")


# In[94]:


from platypus import NSGAII, SPEA2, IBEA, Problem, Integer

problem_types = []
for i in range(len(df.index)):
    problem_types = np.append(problem_types, Integer(0, ncomb-1))


# In[105]:


st.header('Optimização')
#up = st.selectbox("Tipo de renovação", ["Paredes", "Janelas", "Fonte da sua energia"])
option = st.radio('Selecione o algoritmo de optimização:',
                  [#"No optimization",
                   'NSGAII',
                   'SPEA2',
                   'IBEA'])
n = st.number_input("Número de combinações a explorar pelo algoritmo", 50, 50000, 100)
st.write("---")


# In[106]:


problem = Problem(len(df.index), 3)
problem.types[:] = problem_types
problem.function = opt_df


# In[107]:


def option_opt(option):
        if option == "NSGAII":
            algorithm = NSGAII(problem)
            algorithm.run(n)
            x = [s.objectives[0] for s in algorithm.result]
            y = [s.objectives[1] for s in algorithm.result]
            z = [s.objectives[2] for s in algorithm.result]
            return x, y, z
        if option == "SPEA2":
            algorithm = SPEA2(problem)
            algorithm.run(n)
            x = [s.objectives[0] for s in algorithm.result]
            y = [s.objectives[1] for s in algorithm.result]
            z = [s.objectives[2] for s in algorithm.result]
            return x, y, z
        if option == "IBEA":
            algorithm = IBEA(problem)
            algorithm.run(n)
            x = [s.objectives[0] for s in algorithm.result]
            y = [s.objectives[1] for s in algorithm.result]
            z = [s.objectives[2] for s in algorithm.result]
            return x, y, z
        else:
            return print("No optimization selected")


# In[108]:


results = option_opt(option)


# In[109]:


results_df = pd.DataFrame(results).transpose()
results_df.columns = ["Energy Consumption kWh/m2", "Standar Deviation kWh/m2", "total cost €"]


# In[110]:


results_df


# In[111]:


# x_IBEA = [s.objectives[0] for s in algorithm_IBEA.result]
# y_IBEA = [s.objectives[1] for s in algorithm_IBEA.result]
# z_IBEA = [s.objectives[2] for s in algorithm_IBEA.result]

# IBEA_df = pd.DataFrame([x_IBEA, y_IBEA, z_IBEA]).transpose()
# IBEA_df.columns = ["Energy Consumption kWh/m2", "STD", "total cost €"]


# In[112]:


import plotly.graph_objects as go
fig = go.Figure()



fig = fig.add_trace(go.Scatter3d(x = results_df["Energy Consumption kWh/m2"],
                                 y = results_df["Standar Deviation kWh/m2"],
                                 z = results_df["total cost €"],
                                 opacity = 0.5,
                                 mode = "markers",
                                 name = option,
                                 marker = dict(size = 4),
                                 #alphahull = 0,
                                 showlegend= True))
fig.add_trace(go.Scatter3d(x = [opt_df(np.repeat(0, 20))[0]],
                            y = [opt_df(np.repeat(0, 20))[1]],
                            z = [opt_df(np.repeat(0, 20))[2]],
                            name = i,
                            opacity = 0.5,
                            mode = "markers",
                            marker = dict(size = 0),
                            #alphahull = -1,
                            showlegend= True))

st.plotly_chart(fig, use_container_width=True, sharing="streamlit")


# In[113]:


fig.show()


# In[ ]:




