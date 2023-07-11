#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required packages from python library
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'


# In[2]:


data = pd.read_csv("data.csv").drop("Unnamed: 0", axis=1)


# In[3]:


X = data.drop(["R", "Ntc Valor", "Nic Valor", "Nvc Valor", "EPC", "TARGET", "Ntc Limite", "walls_u", "roofs_u", "floors_u", "window_u"], axis=1)
y = data[["R", "Ntc Valor", "Nic Valor", "Nvc Valor", "Ntc Limite"]]


# In[4]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
model = ExtraTreesRegressor(n_jobs=-1, random_state=42)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y["Ntc Valor"], test_size=0.33, random_state=42)


# In[6]:


with st.spinner("""This is an early design stage simulator and does not represent accurate design execution stage simulations"""):
    @st.cache_resource()  # 👈 Added this
    def ntc_():
        return model.fit(X_train, y_train)
    
et_ntc = ntc_()


# In[7]:


preds = et_ntc.predict(X_test)


# In[8]:


def period_to_epoch(x):
    if pd.isna(x) == True:
        return "is null"
    if x == "before 1918":
        return 0
    if x == "between 1919 and 1945":
        return 1
    if x== "between 1946 andnd 1960":
        return 2
    if x== "between 1961 and 1970":
        return 3
    if x== "between 1971 and 1980":
        return 4
    if x== "between 1981 and 1990":
        return 5
    if x== "between 1991 and 1995":
        return 6
    if x== "between 1996 and 2000":
        return 7
    if x== "between 2001 and 2005":
        return 8
    else:
        return 9


# In[9]:


def epochs_to_period(x):
    if x == 0:
        return "before 1918"
    if x == 1:
        return "between 1919 and 1945"
    if x== 2:
        return "between 1946 and 1960"
    if x== 3:
        return "between 1961 and 1970"
    if x== 4:
        return "between 1971 and 1980"
    if x== 5:
        return "between 1981 and 1990"
    if x== 6:
        return "between 1991 and 1995"
    if x== 7:
        return "between 1996 and 2000"
    if x== 8:
        return "between 2001 and 2005"
    else:
        return "Posterior and 2005"


# In[10]:


period_df = pd.DataFrame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
period_df["label"] = period_df[0].apply(epochs_to_period)
period_df.columns=["Code", "Period of Construction"]


# In[11]:


typology_type = ['> T6', 'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6']
typology_labels = [0, 1, 2, 3, 4, 5, 6, 7]
typology_df = pd.DataFrame([typology_labels, typology_type]).T.apply(np.roll, shift=-1)


# In[12]:


epc_type = ['Building', 'Fraction (without horizontal property)', 'Fraction (horizontal property)']
epc_type_labels = [0,1, 2]
epc_type_df = pd.DataFrame([epc_type_labels, epc_type]).T
epc_type_df.columns = ["Code", "Tipo de Imóvel"]


# In[13]:


district_types = pd.read_csv("disctrict_types.csv").drop("Unnamed: 0", axis=1)
district_types.columns = ["Code", "Distrito"]


# In[14]:


wall_types = pd.read_csv("wall_types.csv")
roof_types = pd.read_csv("roof_types.csv")
floor_types = pd.read_csv("floors_types.csv")
window_types = pd.read_csv("window_types.csv")


# In[15]:


ac_sources = pd.read_csv("ac_sources.csv").iloc[:12]
ac_types = pd.read_csv("ac_types.csv").iloc[:16]

dhw_sources = pd.read_csv("dhw_sources.csv")
dhw_types = pd.read_csv("dhw_types.csv")


# In[16]:


sample = X_test.sample(10).reset_index(drop=True)


# In[17]:


st.write("""
# Lisbon Building energy use optimizer for public policies

This app allows the user to upload a csv file with multiple buildings and their respective characteristics and optimize their retrofit.

""")
st.write("---")


# In[18]:


@st.cache_data
def convert_example(df):
    return df.to_csv().encode('utf-8')

example = convert_example(sample)

st.download_button(
    label="Download template input .csv",
    data=example,
    file_name='template_upload.csv',
    mime='text/csv',)

st.dataframe(sample)
st.write("---")


# In[19]:


st.write("""
## Column codes for building features:
Here you can check the respective code number for each categorical building feature
""")


# In[20]:


col_district, col_type, col_typology = st.columns(3)

col_district.write(""" ### Distrito""")
col_district.dataframe(district_types)

col_type.write(""" ### Tipo de Imóvel""")
col_type.dataframe(epc_type_df)

col_typology.write(""" ### Tipologia""")
col_typology.dataframe(typology_df)


# In[21]:


wall_df = wall_types[["Tipo de Solução", "Solution"]]
wall_df.columns = ["Code", "Wall Solution"]


# In[22]:


roof_df = roof_types[["Tipo de Solução", "Solution"]]
roof_df.columns = ["Code", "Roof Solution"]


# In[23]:


col_epoch, col_walls, col_roofs = st.columns(3)

col_epoch.write(""" ### epoch""")
col_epoch.dataframe(period_df)

col_walls.write(""" ### walls_type""")
col_walls.dataframe(wall_df)

col_roofs.write(""" ### roofs_type""")
col_roofs.dataframe(roof_df)


# In[24]:


floor_df = floor_types[["Tipo de Solução", "solution"]]
floor_df.columns = ["Code", "Floor Solution"]


# In[25]:


window_df = window_types[["labels", "Tipo de Solução 1"]]
window_df.columns = ["Code", "Window Solution"]


# In[26]:


ac_sources.columns= ["Code", "Climatization Energy Source"]


# In[27]:


col_floors, col_window, col_acs = st.columns(3)

col_floors.write(""" ### floors_type""")
col_floors.dataframe(floor_df)

col_window.write(""" ### window_type""")
col_window.dataframe(window_df)

col_acs.write(""" ### ac_source""")
col_acs.dataframe(ac_sources)


# In[28]:


ac_types.columns = ["Code", "Climatization equipment"]


# In[29]:


dhw_sources.columns = ["Code", "Domestic Hot Water energy source"]


# In[30]:


dhw_types.columns = ["Code", "Domestic Hot Water equipment"]


# In[31]:


col_ace, col_dhws, col_dhwe = st.columns(3)

col_ace.write(""" ### ac_equipment""")
col_ace.dataframe(ac_types)

col_dhws.write(""" ### dhw_source""")
col_dhws.dataframe(dhw_sources)

col_dhwe.write(""" ### dhw_equipment""")
col_dhwe.dataframe(dhw_types)


# In[32]:


st.write("""
## Building data upload

Here you can upload the .csv file filled in as shown in "template_upload.csv", but with your buildings
""")


# In[291]:


upload = st.file_uploader("Input CSV file with all the buildings you want to optimize")


# In[34]:


@st.cache_data
def read_csv(csv):
    sampl_up = pd.read_csv(csv).drop("Unnamed: 0", axis=1)
    return sampl_up

if upload != None:
    sample = read_csv(upload)
    preds = et_ntc.predict(sample)
else:
    preds = et_ntc.predict(sample)


# In[35]:


col_a, col_b, col_c = st.columns(3)
simulate_button = col_b.button('Predict annual energy loads')

import plotly.express as px
if simulate_button == True:

    df = pd.DataFrame(preds)
    df.columns = ["Energy (kWh/sqm)"]
    fig = px.histogram(df, x="Energy (kWh/sqm)", color_discrete_sequence=['indianred'])
    st.plotly_chart(fig)
    col_a1, col_c1 = st.columns(2)
    col_a1.metric("Average building Energy needs:", str(round(np.average(preds), 2)) + " kWh/sqm")
    col_c1.metric("Standard deviation:", str(round(np.std(preds), 2)) + " kWh/sqm")
    #fig.show()


# In[36]:


retrofits = ["Wall insulation (EPS)", 
            "Floor insulation", 
            "Roof insulation (EPS)", 
            "Window replacement (PVC)", 
            "Air-to-water pump",
            "Efficient AC units", 
            "Solar panels for DHW", "Solar panels for energy production"]


# In[37]:


st.write("---")
st.write(""" ## Optimization

Here you can define the optimization problem variables, algorithm, and their parameters""")

st.subheader("Variables")
variables = st.multiselect("Select Retrofits from government's retrofit available funding list", 
                            retrofits, default=retrofits)


# In[174]:


st.subheader("Retrofit costs")

if "Wall insulation (EPS)" in variables:
    wall_cost = st.number_input("Wall retrofit cost (€/sqm)", 0, 500, 70)

if "Floor insulation" in variables:
    floor_cost = st.number_input("Floor retrofit cost (€/sqm)", 0, 500, 30)
    
if "Roof insulation (EPS)" in variables:
    roof_cost = st.number_input("Roof retrofit cost (€/sqm)", 0, 500, 30)

if "Window replacement (PVC)" in variables:
    window_cost = st.number_input("Window retrofit cost (€/sqm)", 0, 500, 300)
    
if "Air-to-water pump" in variables:
    pump_cost = st.number_input("Air-to-water heat pump retrofit cost (€/unit)", 0, 10000, 4000)

if "Efficient AC units" in variables:
    ac_cost = st.number_input("Efficient AC units retrofit cost (€/unit)", 0, 2000, 700)
    
if "Solar panels for DHW" in variables:
    solar_dhw_cost = st.number_input("Solar panels for DHW retrofit cost (€/unit)", 0, 10000, 2000)
    
if "Solar panels for energy production" in variables:
    solar_energy_cost = st.number_input("Solar panels for energy production retrofit cost (€/unit)", 0, 2000, 600)
    
st.write("---")


# In[175]:


from platypus import *


# In[176]:


platypus_vars = []
vars_columns = []
for j in sample.index:
    for i in variables:
        platypus_vars.append(Integer(0, 1))
        vars_columns.append(i + " Building " + str(j + 1))

vars_columns.append("Average Energy Needs [kWh/sqm]")
vars_columns.append("Standard Deviation [kWh/sqm]") 
vars_columns.append("Cost [€]")  


# In[177]:


vars_df = pd.DataFrame(np.array(platypus_vars).reshape(len(sample), len(variables)))


# In[178]:


vars_df.columns = variables


# In[179]:


from time import time


# In[229]:


def buildings_opt(x):
    #start = time()
    sampl = sample.copy()
    vars_df = pd.DataFrame(np.array(x).reshape(len(sampl), len(variables)))
    vars_df.columns = variables
    costs = [0]
    
    if "Wall insulation (EPS)" in vars_df.columns:
        for i in enumerate(vars_df["Wall insulation (EPS)"]):
            tipo = sampl["Tipo de Imóvel"].iloc[i[0]]
            npisos = sampl["Número Total de Pisos"].iloc[i[0]]
            wall_area = sampl["walls_area"].iloc[i[0]]
            if i[1] == 1:
                sampl["walls_type"].iloc[i[0]] = 0
                costs.append(wall_cost*wall_area)
                
    if "Floor insulation" in vars_df.columns:
        for i in enumerate(vars_df["Floor insulation"]):
            tipo = sampl["Tipo de Imóvel"].iloc[i[0]]
            npisos = sampl["Número Total de Pisos"].iloc[i[0]]
            floor_area = sampl["floors_area"].iloc[i[0]]
            if i[1] == 1:
                sampl["floors_type"].iloc[i[0]] = 4
                if tipo == 0:
                    costs.append(floor_cost*floor_area*npisos)
                else:
                    costs.append(floor_cost*floor_area)
                    
                
    if "Roof insulation (EPS)" in vars_df.columns:
        for i in enumerate(vars_df["Roof insulation (EPS)"]):
            roof_area = sampl["roofs_area"].iloc[i[0]]
            if i[1] == 1:
                sampl["roofs_type"].iloc[i[0]] = 0
                costs.append(roof_cost*roof_area)
    
    if "Window replacement (PVC)" in vars_df.columns:
        for i in enumerate(vars_df["Window replacement (PVC)"]):
            window_area = sampl["window_area"].iloc[i[0]]
            tipo = sampl["Tipo de Imóvel"].iloc[i[0]]
            npisos = sampl["Número Total de Pisos"].iloc[i[0]]
            if i[1] == 1:
                sampl["window_type"].iloc[i[0]] = 9
                costs.append(window_cost*window_area)

    
    if "Air-to-water pump" in vars_df.columns:
        for i in enumerate(vars_df["Air-to-water pump"]):
            npisos = sampl["Número Total de Pisos"].iloc[i[0]]
            tipo = sampl["Tipo de Imóvel"].iloc[i[0]]
            if i[1] == 1:
                sampl["dhw_equipment"].iloc[i[0]] = 2
                sampl["ac_equipment"].iloc[i[0]] = 4
                if tipo == 0:
                    sampl["nr_dhw_units"].iloc[i[0]] = npisos
                    costs.append(pump_cost*npisos)  
                else:
                    sampl["nr_dhw_units"].iloc[i[0]] = 1
                    costs.append(pump_cost)

                
    if "Efficient AC units" in vars_df.columns:
        for i in enumerate(vars_df["Efficient AC units"]):
            npisos = sampl["Número Total de Pisos"].iloc[i[0]]
            tipologia = sampl["Tipologia"].iloc[i[0]]
            tipo = sampl["Tipo de Imóvel"].iloc[i[0]]
            if i[1] == 1:
                sampl["ac_equipment"].iloc[i[0]] = 4
                if tipo == 0:
                    if tipologia == 0:
                        sampl["nr_ac_units"].iloc[i[0]] = npisos*8
                        costs.append(ac_cost*8*npisos)
                        sampl["nr_ac_units"].iloc[i[0]] = npisos*(tipologia)
                    else:
                        sampl["nr_ac_units"].iloc[i[0]] = tipologia*npisos
                        costs.append(ac_cost*npisos*tipologia)
                else:
                    if tipologia == 0:
                        sampl["nr_ac_units"].iloc[i[0]] = 8
                        costs.append(ac_cost*8*0.15)
                        
                    else:
                        sampl["nr_ac_units"].iloc[i[0]] = tipologia
                        costs.append(ac_cost*tipologia)
                    
    
    if "Solar panels for DHW" in vars_df.columns:
        for i in enumerate(vars_df["Solar panels for DHW"]):
            npisos = sampl["Número Total de Pisos"].iloc[i[0]]
            tipologia = sampl["Tipologia"].iloc[i[0]]
            tipo = sampl["Tipo de Imóvel"].iloc[i[0]]
            if i[1] == 1:
                sampl["dhw_equipment"].iloc[i[0]] = 4
                sampl["dhw_source"].iloc[i[0]] = 10
                if tipo == 0:
                    sampl["nr_dhw_units"].iloc[i[0]] = npisos
                    costs.append(solar_dhw_cost*npisos)
                else:
                    sampl["nr_dhw_units"].iloc[i[0]] = 1
                    costs.append(solar_dhw_cost)
                
    if "Solar panels for energy production" in vars_df.columns:
        for i in enumerate(vars_df["Solar panels for energy production"]):
            npisos = sampl["Número Total de Pisos"].iloc[i[0]]
            tipologia = sampl["Tipologia"].iloc[i[0]]
            tipo = sampl["Tipo de Imóvel"].iloc[i[0]]
            if i[1] == 1:
                sampl["ac_source"].iloc[i[0]] = 10
                sampl["dhw_source"].iloc[i[0]] = 10
                if tipo == 0:
                    if tipologia == 0:
                        costs.append(solar_energy_cost*8)
                    else:
                        costs.append(solar_energy_cost*tipologia*npisos)
                else:
                    if tipologia == 0:
                        costs.append(solar_energy_cost*8)
                    else:
                        sampl["nr_ac_units"].iloc[i[0]] = tipologia
                        sampl["nr_dhw_units"].iloc[i[0]] = 1
                        costs.append(solar_energy_cost*tipologia)
                                         
    cost = np.sum(costs)
    preds = et_ntc.predict(sampl)
    avg = np.average(preds)
    std = np.std(preds)
    #end = time()
    return [avg, std, cost]          


# In[230]:


algs = ["NSGAII", "NSGAIII", "SPEA2", "IBEA"]


# In[231]:


st.subheader("Select Optimization Algorithm")
alg = st.selectbox("Choose the algorithm you want to test", algs, 2)

problem = Problem(len(platypus_vars), 3)
problem.types[:] = platypus_vars
problem.function = buildings_opt
itrs = st.number_input("Number of iterations you wish to test:", 200, value=500)
if alg == "NSGAII":
    pop = round(itrs/50)
    sol = pop
    algorithm = NSGAII(problem, population_size=pop)

if alg == "SPEA2":
    pop = round(itrs/50)
    algorithm = SPEA2(problem, population_size=pop)

if alg == "NSGAIII":
    divisions = 20
    algorithm = NSGAIII(problem, divisions_outer=20)

if alg == "IBEA":
    pop = round(itrs/50)
    algorithm = IBEA(problem, population_size=pop)


# In[271]:


col_empty, col_button, col_empty = st.columns(3)
opt = col_button.button("Start Optimization!")


# In[288]:


if opt == True:
    with st.spinner("The increase in the number of iterations increases the execution time. Each iteration takes around 0.02 seconds to complete."):
        algorithm.run(itrs)
        nondominated_solutions = nondominated(algorithm.result)


# In[279]:


if opt == True:
    global_arr= []
    for s in nondominated_solutions:
        for v in enumerate(platypus_vars):
            if "Integer" in str(v[1]):
                global_arr.append(v[1].decode(s.variables[v[0]]))
            else:
                global_arr.append(round(s.variables[v[0]], 2))
        global_arr.append(s.objectives[0])
        global_arr.append(s.objectives[1])
        global_arr.append(s.objectives[2])
    final = np.array(global_arr).reshape(len(nondominated_solutions), len(platypus_vars) +3)
    final_df = pd.DataFrame(final)
    final_df.columns = vars_columns


# In[280]:


import plotly.express as px
if opt == True:                         
    fig = px.scatter_3d(final_df, 
                        x="Average Energy Needs [kWh/sqm]", 
                        y="Standard Deviation [kWh/sqm]", 
                        z="Cost [€]",
                        color="Average Energy Needs [kWh/sqm]",
                        color_continuous_scale=px.colors.diverging.Tealrose)

    fig.update_layout(
                        autosize=True,
                        width=1200,
                        height=800)
    camera = dict(
        up=dict(x=0, y=0, z=1.25),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=-1.25, z=1)
    )

    fig.update_layout(scene_camera=camera)
    st.plotly_chart(fig)
    #fig.show()


# In[281]:


def binary_convert(x):
    if x == 1.0:
        return True
    else:
        return False


# In[286]:


if opt == True:
    for i in final_df.drop(["Average Energy Needs [kWh/sqm]", "Standard Deviation [kWh/sqm]", "Cost [€]"],axis=1):
        final_df[i] = final_df[i].apply(lambda x: binary_convert(x))
        download_df = final_df.T
        download_df.columns = ["Solution " + str(i + 1) for i in final_df.index]


# In[260]:


if opt == True:
    col_h, col_opt_down, col_i = st.columns(3)
    @st.cache_data
    def convert_example(df):
        return df.to_csv().encode('utf-8')

    example = convert_example(download_df)

    col_opt_down.download_button(label="Download optimum solutions as CSV",
                                 data=example,
                                 file_name='optimum_solutions.csv',
                                 mime='text/csv',)

    st.write("---")


# In[262]:





# In[264]:





# In[ ]:




