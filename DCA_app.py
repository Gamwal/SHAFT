# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 08:14:18 2022

@author: gamaliel.adun
"""
#%% Import the needed Libraries & Equations
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import plotly.express as px
from PIL import Image
import altair as alt
import streamlit as st
from datetime import date
from arps_equations import *
from data_loader import *
from display_plots import *

#%%
image = Image.open('SHAFT_logo.jpg')
st.image(image, use_column_width=True)

#%%% Welcome text and 
st.write(""" 
         ### Sahara Hydocarbon Assets Forecasting Tool (S.H.A.F.T)
         #### Production Analysis and Forecasting using Decline Curve Analysis         
         
         """)

#%% Data import
uploaded_file = st.file_uploader("Choose a file",type=['xlsx','xls'])
if (uploaded_file == None):
    st.stop()

data = loader_excel(uploaded_file)

#datafile = 'C:/Users/adun.gamaliel/Downloads/data.mdb'
#data = read_access_tables(datafile)

with st.expander("Expand to view input table"):
    data

#%%% Select the Domain over which to perform the DCA
dca_method = st.radio('How would you like to carry out your Decline Curve Analysis?',
         ('Well','Reservoir','Field'),horizontal=True)

st.write(f'You selected DCA by {dca_method}')

#%%% Sidebar to select the columns of the data to be used for the DCA
st.sidebar.header('Input selection')
form1 = st.sidebar.form("my_form")

well = form1.selectbox(
    "Select Column for Unique Well Identifier (Must be of the form 'fieldnamewellno:reservoir')",
    (data.columns.to_list()))

oil_prod = form1.selectbox(
    'Select Column for Oil Production',
    (data.columns.to_list()))

prod_date = form1.selectbox(
    'Select Column for Production Date',
    (data.columns.to_list()))

choke = form1.selectbox(
    'Select Column for Choke Size',
    (data.columns.to_list()))

prod_days = form1.selectbox(
    'Select Column for Days of Production',
    (data.columns.to_list()))

gas_prod = form1.selectbox(
    'Select Column for Gas Production',
    (data.columns.to_list()))

water_prod = form1.selectbox(
    'Select Column for Water Production',
    (data.columns.to_list()))

form1.form_submit_button("OK")

if (gas_prod == water_prod):
    st.stop()

#%%% Clean the data and extract just what you need
data = data[[well, prod_date, choke, oil_prod, prod_days, gas_prod, water_prod]]
data1 = data.drop(data[data[oil_prod] == 0].index)
DCA_data = data1.drop(data1[data1[oil_prod] == np.nan].index)

DCA_data[['Well','Well_Reservoir']] = DCA_data[well].str.split(':',expand=True)
DCA_data['Well'] = DCA_data['Well'].str.slice(start=4)
DCA_data['Field'] = DCA_data[well].str.slice(stop=4)
DCA_data['Reservoir'] = DCA_data['Field'].str.cat(DCA_data['Well_Reservoir'],sep='_')
DCA_data['date'] = DCA_data[prod_date].dt.date
DCA_data['month_days'] = DCA_data[prod_date].dt.days_in_month
DCA_data['Oil Rate'] = DCA_data[oil_prod] / DCA_data['month_days']
DCA_data['Gas Rate'] = DCA_data[gas_prod] / DCA_data['month_days']
DCA_data['GOR'] = DCA_data[gas_prod] / DCA_data[oil_prod]
DCA_data['Water Cut'] = DCA_data[water_prod] / (DCA_data[water_prod] + DCA_data[oil_prod])
DCA_data['Well Count'] = 1

del data #to reduce memory usage

#%%% Select the DCA type you intend to use
if  dca_method == 'Well':
    sort_by = st.selectbox('Select Well of interest', (DCA_data[well].unique()))
    subset = DCA_data[DCA_data[well]==sort_by].sort_values(by=choke,axis=0)
    color_1 = choke
    tag = 'Choke Size'
elif dca_method == 'Reservoir':
    sort_by = st.selectbox('Select Reservoir of interest', (DCA_data['Reservoir'].unique()))
    subset = DCA_data[DCA_data['Reservoir']==sort_by].sort_values(by='Reservoir',axis=0)
    color_1 = 'Well'
    tag = 'Well'
elif dca_method == 'Field':
    sort_by = st.selectbox('Select Field of interest', (DCA_data['Field'].unique()))
    subset = DCA_data[DCA_data['Field']==sort_by].sort_values(by='Field',axis=0)
    color_1 = 'Reservoir'
    tag = 'Reservoir'
else:
    st.write('Invalid selection')
    
table = subset.sort_values(by=prod_date)
with st.expander(f"Expand to view table for selected {dca_method}"):
    table
keep = subset #For reference before overwriting begins

#%%% Plot the input data
plot_subset = subset.copy()
plot_subset[choke] = plot_subset[choke].astype(str)
input_plotter(plot_subset,choke,prod_date,color_1,dca_method,sort_by,tag)

plot_subset = subset.copy()
plot_subset[choke] = plot_subset[choke].astype(str)
#fig = px.scatter(plot_subset,x=prod_date,y='Oil Rate',color=color_1,title=f'{dca_method} name: {sort_by}')
#fig.update_layout(margin=dict(l=10, r=10, t=50, b=10),paper_bgcolor="Black")
#fig.update_yaxes(title_text='Oil Rate(bbl/d)',range=[0,subset['Oil Rate'].max()+100])
#fig.update_xaxes(title_text='Production Date')
#fig.update_layout({'legend_title_text': tag})
#st.plotly_chart(fig)

#%% manipulate the data depending on the DCA type to be carried out
if dca_method in ('Reservoir','Field'):
    if subset[prod_date].nunique() == subset.shape[0]:
        st.write('No production summation required')
        form2 = st.form("subsetting_form")
        start_date, end_date = form2.select_slider(
             'Select time window of Analysis',
             options=subset['date'].sort_values().unique(),
             value=(subset['date'].min(), subset['date'].max()))
        #st.write('You selected time window between', start_date, 'and', end_date)
        FC_start_date = form2.slider(
             'Select Forecast Start Date',
             min_value = subset['date'].max(),
             max_value = date.today(),
             value = subset['date'].max(),
             format = "YYYY-MM-DD")
        fc_years = form2.number_input(
            'Enter number of years for Forecast')
        EL = form2.number_input(
            'Enter value for economic limit in bbl/d')
        form2.form_submit_button("OK")
        if (float(fc_years) < 0 or float(fc_years) > 239):
            st.write(f"{fc_years} is out of bounds \n select a valid time")
            st.stop()
        #Final selection for DCA using desired user selected boundaries
        subset = subset.sort_values(prod_date).set_index(prod_date,drop=False)
        subset['Well Count'] = subset.index.value_counts()
        subset.reset_index(drop=True,inplace=True)
        subset['Cum Oil Prod'] = subset[oil_prod].cumsum()
        subset = subset.loc[(subset['date'] >= start_date) & (subset['date'] <= end_date)]
        subset['Cum days'] = (subset[prod_date] - subset[prod_date].min()).dt.days
        subset = subset.sort_values(prod_date)
        choke=None
        choke_size=None
    else:
        st.write("""
                 Production summation was required for overlapping production dates.
                 See new combined plot below""")        
        # Perform the summation of the data for same date events
        # For subset with same prod date, sum the production and prod days then
        # recalulate oilrate cumoil cum days, cum years and add create a new
        # subset dataframe using a for loop to loop through unique days
        subset = subset.sort_values(prod_date).set_index(prod_date,drop=False)
        subset['Well Count'] = subset.index.value_counts()
        subset[oil_prod] = subset.groupby(subset.index)[oil_prod].sum()
        subset[gas_prod] = subset.groupby(subset.index)[gas_prod].sum()
        subset[water_prod] = subset.groupby(subset.index)[water_prod].sum()
        #subset['delta'] = subset[prod_date] - subset[prod_date].min()
        subset['Cum days'] = (subset[prod_date] - subset[prod_date].min()).dt.days
        subset['month_days'] = subset[prod_date].dt.days_in_month
        subset['Oil Rate'] = subset[oil_prod] / subset['month_days']
        subset['Gas Rate'] = subset[gas_prod] / subset['month_days']
        subset['Water Cut'] = subset[water_prod] / (subset[water_prod] + subset[oil_prod])
        subset['Cum Years'] = subset['Cum days'] / 365
        subset['test'] = subset.index == subset[prod_date]
        subset.reset_index(drop=True,inplace=True)
        subset.drop_duplicates(prod_date,inplace=True)
        subset['Cum Oil Prod'] = subset[oil_prod].cumsum()
        keep = subset
        
        #plot the data the new data
        combined_production_plotter(subset,prod_date,color_1,dca_method,sort_by,tag)
        #fig = px.scatter(subset,x=prod_date,y='Oil Rate',color=None,title=f'{dca_method} name: {sort_by}')
        #fig.update_layout(margin=dict(l=10, r=10, t=50, b=10),paper_bgcolor="Black")
        #fig.update_yaxes(title_text='Oil Rate(bbl/d)',range=[0,subset['Oil Rate'].max()*1.1])
        #fig.update_xaxes(title_text='Production Date')
        #fig.update_layout({'legend_title_text': tag})
        #st.plotly_chart(fig)
                
        form2 = st.form("subsetting_form")
        start_date, end_date = form2.select_slider(
             'Select time window of Analysis',
             options=subset['date'].sort_values().unique(),
             value=(subset['date'].min(), subset['date'].max()))
        #st.write('You selected time window between', start_date, 'and', end_date)
        FC_start_date = form2.slider(
             'Select Forecast Start Date',
             min_value = subset['date'].max(),
             max_value = date.today(),
             value = subset['date'].max(),
             format = "YYYY-MM-DD")        
        fc_years = form2.number_input(
            'Enter number of years for Forecast')
        EL = form2.number_input(
            'Enter value for economic limit in bbl/d')
        form2.form_submit_button("OK")
        if (float(fc_years) < 0 or float(fc_years) > 239):
            st.write(f"{fc_years} is invalid \n select a valid time")
            st.stop()
        #Final selection for DCA using desired user selected boundaries
        subset = subset.loc[(subset['date'] >= start_date) & (subset['date'] <= end_date)]
        subset = subset.drop(subset[subset['Oil Rate'] == 0].index)
        subset['Cum days'] = (subset[prod_date] - subset[prod_date].min()).dt.days
        subset = subset.sort_values(prod_date)
        choke=None
        choke_size=None
else:
    subset['Cum Oil Prod'] = subset[oil_prod].groupby(subset[well]).cumsum()
    subset['Cum days'] = subset[prod_days].groupby(subset[well]).cumsum()
    subset['Cum Years'] = subset[prod_days].groupby(subset[well]).cumsum() / 365
    subset = subset.sort_values(by=choke)
    subset[choke] = subset[choke].astype(str)
    form2 = st.form("subsetting_form")
    choke_size = form2.select_slider(
         'Select a choke size for analysis',
         options=subset[choke].unique())
    #st.write('Selected Choke Size is', choke_size)
    start_date, end_date = form2.select_slider(
         'Select time window of Analysis',
         options=subset['date'].sort_values().unique(),
         value=(subset['date'].min(), subset['date'].max()))
    #st.write('You selected time window between', start_date, 'and', end_date)
    FC_start_date = form2.slider(
         'Select Forecast Start Date',
         min_value = subset['date'].max(),
         max_value = date.today(),
         value = subset['date'].max(),
         format = "YYYY-MM-DD")
    fc_years = form2.number_input(
        'Enter number of years for Forecast')
    EL = form2.number_input(
        'Enter value for economic limit in bbl/d')
    form2.form_submit_button("OK")
    if (float(fc_years) < 0 or float(fc_years) > 239):
        st.write(f"{fc_years} is out of bounds, select a valid time")
        st.stop()
    #Final selection for DCA using desired user selected boundaries
    subset = subset.sort_values(prod_date).set_index(prod_date,drop=False)
    subset['Well Count'] = subset.index.value_counts()
    subset.reset_index(drop=True,inplace=True)
    subset = subset.loc[(subset[choke] == choke_size) & (subset['date'] >= start_date) & (subset['date'] <= end_date)]
    subset = subset.drop(subset[subset['Oil Rate'] == 0].index)
    subset['Cum days'] = (subset[prod_date] - subset[prod_date].min()).dt.days
    subset = subset.sort_values(prod_date)
    
#%%%
spacing = 200
#subset = subset.drop(subset[subset['Oil Rate'] == 0].index)
xdata = np.array(subset['Cum days'])
ydata = np.array(subset['Oil Rate'])

qi =  subset['Oil Rate'].max()
exp_qi = np.log(qi)

lBounds = [qi-0.01,0,0.01]
uBounds = [qi,0.01,0.99]
bounds = (lBounds,uBounds)

try:
    ExpPara, ArpsRateExpMin = curve_fit(ArpsRateExp,xdata,np.log(ydata),bounds=([exp_qi-0.01,0],[exp_qi,0.001]),method='trf')
    HarPara, ArpsRateHarMin = curve_fit(ArpsRateHar,xdata,ydata,bounds=([qi-0.01,0],[qi,0.01]),method='trf')
    HypPara, ArpsRateHypMin = curve_fit(ArpsRateHyp,xdata,ydata,bounds=bounds,method='trf')
except ValueError:
    st.write("Invalid selection, please check the choke details")
    st.stop()

#ExpPara, ArpsRateExpMin = curve_fit(ArpsRateExp,xdata,np.log(ydata),bounds=([exp_qi-0.01,0],[exp_qi,0.001]),method='trf')
#HarPara, ArpsRateHarMin = curve_fit(ArpsRateHar,xdata,ydata,bounds=([qi-0.01,0],[qi,0.01]),method='trf')
#HypPara, ArpsRateHypMin = curve_fit(ArpsRateHyp,xdata,ydata,bounds=bounds,method='trf')

fc_days = fc_years*365
true_cumm = (date.today() - subset['date'].min()).days

k = keep['date'].max()
f = (FC_start_date - k)
#length2 = np.linspace(subset['Cum days'].max(),true_cumm+fc_days,spacing)
length2 = np.linspace(0,fc_days,spacing)

#dates2 = pd.date_range(start=subset[prod_date].max(),end=date.today()+pd.DateOffset(days=fc_days),periods=spacing).date
#dates2 = pd.date_range(start=DCA_data[prod_date].max()+f,end=DCA_data[prod_date].max()+f+pd.DateOffset(days=fc_days),periods=spacing).date
dates2 = pd.date_range(start=FC_start_date, end=FC_start_date+pd.DateOffset(days=fc_days), periods=spacing).date

FC_Qi = table['Oil Rate'].iloc[-1]

Qi_1, Di_1 = np.exp(ExpPara[0]), ExpPara[1]
#Di_1 = 0.075/100
subset['ArpsRateExpPredict'] = Qi_1 * np.exp(-Di_1*subset['Cum days'])
#ArpsRateExpFC = Qi_1 * np.exp(-1*Di_1*length2)
ArpsRateExpFC = FC_Qi * np.exp(-1*Di_1*length2)
ArpsRateExpFC = [x if x >= EL else np.nan for x in ArpsRateExpFC]

Qi_2, Di_2 = HarPara[0], HarPara[1]
subset['ArpsRateHarPredict'] = Qi_2/(1+(Di_2*subset['Cum days']))
#ArpsRateHarFC = Qi_1/(1+(Di_2*length2))
ArpsRateHarFC = FC_Qi /(1+(Di_2*length2))
ArpsRateHarFC = [x if x >= EL else np.nan for x in ArpsRateHarFC]

Qi_3, Di_3, b_3 = HypPara[0], HypPara[1], HypPara[2]
subset['ArpsRateHypPredict'] = Qi_3 / ((1+(b_3*Di_3*subset['Cum days']))**(1/b_3))
#ArpsRateHypFC = Qi_1  / ((1+(b_3*Di_3*length2))**(1/b_3))
ArpsRateHypFC = FC_Qi / ((1+(b_3*Di_3*length2))**(1/b_3))
ArpsRateHypFC = [x if x >= EL else np.nan for x in ArpsRateHypFC]

if fc_years == 0:
    ArpsRateExpFC = np.nan
    ArpsRateHarFC = np.nan
    ArpsRateHypFC = np.nan

subset_pred_melt = subset.melt(id_vars=prod_date, value_vars=['ArpsRateExpPredict','ArpsRateHarPredict','ArpsRateHypPredict'])
subset_fc = pd.DataFrame({prod_date:dates2, 'ArpsRateExpFC':ArpsRateExpFC, 'ArpsRateHarFC':ArpsRateHarFC, 'ArpsRateHypFC':ArpsRateHypFC})
subset_fc_melt = subset_fc.melt(id_vars=prod_date,value_vars=['ArpsRateExpFC','ArpsRateHarFC','ArpsRateHypFC'])

Qc = subset['Oil Rate'].iloc[-1]
Qel = EL

D_1 = NominalDecline(Di_1, subset['Cum days'].max(), 0)
D_2 = NominalDecline(Di_2, subset['Cum days'].max(), 1)
D_3 = NominalDecline(Di_3, subset['Cum days'].max(), b_3)

Reserves_1 = max(0,ArpsCumProd(Qc, Qel, D_1, 0))
Reserves_2 = max(0,ArpsCumProd(Qc, Qel, D_2, 1))
Reserves_3 = max(0,ArpsCumProd(Qc, Qel, D_3, b_3))

cum_prod = subset['Cum Oil Prod'].max()

Total_1 = cum_prod + Reserves_1
Total_2 = cum_prod + Reserves_2
Total_3 = cum_prod + Reserves_3

Qi_1 = "{:.2f}".format(Qi_1)
Qi_2 = "{:.2f}".format(Qi_2)
Qi_3 = "{:.2f}".format(Qi_3)

Di_1 = "{:.5f}".format(Di_1*100)
Di_2 = "{:.5f}".format(Di_2*100)
Di_3 = "{:.5f}".format(Di_3*100)

b_3 = "{:.2f}".format(b_3)
Qc = "{:.2f}".format(Qc)
Qel = "{:.2f}".format(Qel)

D_1 = "{:.5f}".format(D_1*100)
D_2 = "{:.5f}".format(D_2*100)
D_3 = "{:.5f}".format(D_3*100)

cum_prod = "{:.3f}".format(cum_prod/1000000)

Reserves_1 = "{:.3f}".format(Reserves_1/1000000)
Reserves_2 = "{:.3f}".format(Reserves_2/1000000)
Reserves_3 = "{:.3f}".format(Reserves_3/1000000)

Total_1 = "{:.3f}".format(Total_1/1000000)
Total_2 = "{:.3f}".format(Total_2/1000000)
Total_3 = "{:.3f}".format(Total_3/1000000)

#%%%
#Fit Plot
fit_plotter(subset,prod_date,color_1,dca_method,sort_by,tag,subset_pred_melt)

#Forecast Plot
#forecast_plotter(subset,choke,prod_date,color_1,dca_method,sort_by,tag,subset_fc_melt,EL)

#Combination Plot
combined_plotter(keep,prod_date,color_1,dca_method,sort_by,tag,subset_fc_melt,EL,subset_pred_melt,plot_subset)

combined_log_plotter(keep,prod_date,color_1,dca_method,sort_by,tag,subset_fc_melt,EL,subset_pred_melt,plot_subset)

#%%
DCA_data_sub = keep[['date','Oil Rate']]
DCA_data_sub = DCA_data_sub.loc[(DCA_data_sub['date'] < start_date) | (DCA_data_sub['date'] > end_date)]
DCA_data_sub['Exp'] = np.nan
DCA_data_sub[prod_date] = DCA_data_sub['date']
DCA_data_sub = DCA_data_sub[[prod_date,'Oil Rate','Exp']]

#%%

plot_data = subset[['date','Oil Rate','ArpsRateExpPredict']]
plot_data['Exp'] = plot_data['ArpsRateExpPredict']
plot_data[prod_date] = plot_data['date']
plot_data = plot_data[[prod_date,'Oil Rate','Exp']]

fc_new = subset_fc.copy()
fc_new['Oil Rate'] = np.nan
fc_new['Exp'] = fc_new['ArpsRateExpFC']
fc_new = fc_new[[prod_date,'Oil Rate','Exp']]

new_plot_data = pd.concat([DCA_data_sub,plot_data,fc_new]).reset_index(drop=True).sort_values(prod_date)
new_plot_data['Oil Rate'] = [x if x > 0 else np.nan for x in new_plot_data['Oil Rate']]

rate_max = new_plot_data["Oil Rate"].max() + 10000

#%%

base = alt.Chart(new_plot_data).encode(
    alt.X(prod_date, axis=alt.Axis(title='Date'))
)

line = base.mark_line(stroke='red', interpolate='monotone', point=alt.OverlayMarkDef(color="red")).encode(
    alt.Y('Oil Rate', scale=alt.Scale(type="log", domain=(0.001,rate_max)), axis=alt.Axis(title='Oil Rate', titleColor='red')),tooltip='Oil Rate'
).interactive()

line2 = base.mark_line(stroke='green', interpolate='basis').encode(
    alt.Y('Exp', scale=alt.Scale(type="log", domain=(0.001,rate_max)))#, axis=alt.Axis(title='Forecast', titleColor='green'))
).interactive()

line2 = base.mark_line(stroke='green', interpolate='basis').encode(
    alt.Y('Exp', scale=alt.Scale(type="log", domain=(0.001,rate_max)))#, axis=alt.Axis(title='Forecast', titleColor='green'))
).interactive()

c = alt.layer(line, line2).resolve_scale(
    y = 'independent'
)

#st.altair_chart(c, use_container_width=True)

#%%
keep.reset_index(drop=True,inplace=True)
keep['Water Cut'] = [x if x > 0 else np.nan for x in keep['Water Cut']]
keep['GOR'] = [x if x > 0 else np.nan for x in keep['GOR']]
keep['Oil Rate'] = [x if x > 0 else np.nan for x in keep['Oil Rate']]

base1 = alt.Chart(keep).encode(
    alt.X(prod_date, axis=alt.Axis(title='Date'))
)

line3 = base1.mark_line(stroke='red', interpolate='monotone', point=alt.OverlayMarkDef(color="red")).encode(
    alt.Y('Oil Rate', scale=alt.Scale(type="log", domain=(0.00001,keep["Oil Rate"].max()+100000)), axis=alt.Axis(title='Oil Rate', titleColor='red'))
).interactive()

line4 = base1.mark_line(stroke='cyan', interpolate='monotone').encode(
    alt.Y('Well Count', scale=alt.Scale(domain=(0,keep['Well Count'].max()+10)), axis=alt.Axis(title='Well Count', titleColor='cyan'))
).interactive()

line5 = base1.mark_point(stroke='yellow', interpolate='monotone', size=5).encode(
    alt.Y('GOR', scale=alt.Scale(type="log", domain=(0.00001,keep['GOR'].max()+1000000)), axis=alt.Axis(title='GOR', titleColor='yellow',orient='right',offset=60))
).interactive()

line6 = base1.mark_line(stroke='green', interpolate='monotone', point=alt.OverlayMarkDef(color="green")).encode(
    alt.Y('Gas Rate', scale=alt.Scale(type="log", domain=(0.0001,keep['Gas Rate'].max()+10000000)), axis=alt.Axis(title='Gas Rate', titleColor='green',orient='left',offset=75))
).interactive()

line7 = base1.mark_point(stroke='blue', interpolate='monotone', size=5).encode(
    alt.Y('Water Cut', scale=alt.Scale(domain=(0,1)), axis=alt.Axis(title='Water Cut', titleColor='blue',orient='left',offset=160))
).interactive()

c1 = alt.layer(line3, line4, line5, line6, line7).resolve_scale(
    y = 'independent'
)

st.altair_chart(c1, use_container_width=True)

#%%%
pred_table = subset[[prod_date,'Oil Rate','ArpsRateExpPredict','ArpsRateHarPredict','ArpsRateHypPredict']].reset_index(drop=True)
with st.expander("Expand to view predictions table"):
    pred_table
    
with st.expander("Expand to view forecast table"):
    subset_fc
    
#%%
columns = ('Exponential','Hyperbolic','Harmonic')
cell_text = {'Qi (bbl/d)':[Qi_1,Qi_3,Qi_2],'Di (%)':[Di_1,Di_3,Di_2],'b':['0.00',b_3,'1.00'],'Qc (bbl/d)':[Qc,Qc,Qc],\
             'Qel (bbl/d)':[Qel,Qel,Qel],'D (%)':[D_1,D_3,D_2],'Start date':[start_date,start_date,start_date],'End date':[end_date,end_date,end_date],\
             'Choke size':[choke_size,choke_size,choke_size],'Cum Prod (MMbbls)':[cum_prod,cum_prod,cum_prod],'Reserves (MMbbls)':[Reserves_1,Reserves_3,Reserves_2],'Total (MMbbls)':[Total_1,Total_3,Total_2]}

table = pd.DataFrame(cell_text,index=columns)
table = table.astype('string')
tableT = table.transpose()

st.write("View the parameters for the Arps Methods in the table below")
st.dataframe(tableT)

#%%
@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

csv = convert_df(pred_table)
csv_1 = convert_df(subset_fc)
    
st.download_button(label='Download the prediction table here',data=csv,mime='text/csv',file_name='prediction_table.csv')
st.download_button(label='Download the forecast table here',data=csv_1,mime='text/csv',file_name='forecast_table.csv')
