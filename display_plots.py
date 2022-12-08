# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:25:28 2022

@author: gamaliel.adun
"""
import plotly.express as px
import streamlit as st
import numpy as np

def input_plotter(df,choke,prod_date,color_1,dca_method,sort_by,tag):
    plot_subset = df.copy()
    plot_subset[choke] = plot_subset[choke].astype(str)
    fig = px.scatter(plot_subset,x=prod_date,y='Oil Rate',color=color_1,title=f'{dca_method} name: {sort_by}')
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10),paper_bgcolor="Black")
    fig.update_yaxes(title_text='Oil Rate(bbl/d)',range=[0,df['Oil Rate'].max()+100])
    fig.update_xaxes(title_text='Production Date')
    fig.update_layout({'legend_title_text': tag})
    return st.plotly_chart(fig)

def combined_production_plotter(df,prod_date,color_1,dca_method,sort_by,tag):
    fig = px.scatter(df,x=prod_date,y='Oil Rate',color=None,title=f'{dca_method} name: {sort_by}')
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10),paper_bgcolor="Black")
    fig.update_yaxes(title_text='Oil Rate(bbl/d)',range=[0,df['Oil Rate'].max()*1.1])
    fig.update_xaxes(title_text='Production Date')
    fig.update_layout({'legend_title_text': tag})
    return st.plotly_chart(fig)
    
def fit_plotter(df,prod_date,color_1,dca_method,sort_by,tag,subset_pred_melt):
    fig = px.scatter(df,x=prod_date,y='Oil Rate',color=None,title=f'{dca_method} name: {sort_by}')
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10),paper_bgcolor="Black")
    fig.update_yaxes(title_text='Oil Rate(bbl/d)',range=[0,df['Oil Rate'].max()*1.1])
    fig.update_xaxes(title_text='Production Date')
    fig1 = px.line(subset_pred_melt, x=prod_date, y='value',color='variable')
    fig.add_trace(fig1.data[0])
    fig.add_trace(fig1.data[1])
    fig.add_trace(fig1.data[2])
    return st.plotly_chart(fig)

def forecast_plotter(df,choke,prod_date,color_1,dca_method,sort_by,tag,subset_fc_melt,EL):
    fig4 = px.scatter(df,x=prod_date,y='Oil Rate',color=choke,title=f'{dca_method} name: {sort_by}')
    fig4.update_layout(margin=dict(l=10, r=10, t=50, b=10),paper_bgcolor="Black")
    fig4.update_yaxes(title_text='Oil Rate(bbl/d)',range=[0,df['Oil Rate'].max()*1.1])
    fig4.update_xaxes(title_text='Production Date')
    fig3 = px.line(subset_fc_melt,x=prod_date,y='value',color='variable',line_dash="variable",line_dash_sequence=('dash','dash','dash'))
    fig4.add_trace(fig3.data[0])
    fig4.add_trace(fig3.data[1])
    fig4.add_trace(fig3.data[2])
    fig4.add_hrect(y1=EL,y0=0,fillcolor='red',opacity=0.2,line_color='red',line_width=5, line_dash="dash",annotation_text="Economic Limit")
    return st.plotly_chart(fig4)

def combined_plotter(df,prod_date,color_1,dca_method,sort_by,tag,subset_fc_melt,EL,subset_pred_melt,plot_subset):
    fig1 = px.scatter(df,x=prod_date,y='Oil Rate',color=None,title=f'{dca_method} name: {sort_by}')
    fig1.update_layout(margin=dict(l=10, r=10, t=50, b=10),paper_bgcolor="Black")
    fig1.update_yaxes(title_text='Oil Rate(bbl/d)',range=[0,df['Oil Rate'].max()*1.1])
    fig1.update_xaxes(title_text='Production Date')
    fig2 = px.line(subset_pred_melt, x=prod_date, y='value',color='variable')
    fig1.add_trace(fig2.data[0])
    fig1.add_trace(fig2.data[1])
    fig1.add_trace(fig2.data[2])
    fig3 = px.line(subset_fc_melt,x=prod_date,y='value',color='variable',line_dash="variable",line_dash_sequence=('dash','dash','dash'))
    fig1.add_trace(fig3.data[0])
    fig1.add_trace(fig3.data[1])
    fig1.add_trace(fig3.data[2])
    fig1.add_hrect(y1=EL,y0=0,fillcolor='red',opacity=0.2,line_color='red',line_width=5, line_dash="dash",annotation_text="Economic Limit")
    return st.plotly_chart(fig1)

def combined_log_plotter(df,prod_date,color_1,dca_method,sort_by,tag,subset_fc_melt,EL,subset_pred_melt,plot_subset):
    fig1 = px.scatter(df,x=prod_date,y='Oil Rate',color_discrete_sequence=['red'],title=f'{dca_method} name: {sort_by}',log_y=True)
    fig1.update_layout(margin=dict(l=10, r=10, t=50, b=10),paper_bgcolor="Black")
    fig1.update_yaxes(title_text='Oil Rate(bbl/d)',range=[np.log10(0.01),np.log10(df['Oil Rate'].max()*1.5)])
    fig1.update_xaxes(title_text='Production Date')
    fig2 = px.line(subset_pred_melt, x=prod_date, y='value',color='variable',color_discrete_sequence=['green'])
    fig1.add_trace(fig2.data[0])
    #fig1.add_trace(fig2.data[1])
    #fig1.add_trace(fig2.data[2])
    fig3 = px.line(subset_fc_melt,x=prod_date,y='value',color_discrete_sequence=['green'],line_dash="variable",line_dash_sequence=('dash','dash','dash'))
    fig1.add_trace(fig3.data[0])
    #fig1.add_trace(fig3.data[1])
    #fig1.add_trace(fig3.data[2])
    #fig1.add_hrect(y1=EL,y0=0.01,fillcolor='red',opacity=0.2,line_color='red',line_width=5, line_dash="dash",annotation_text="Economic Limit")
    return st.plotly_chart(fig1)