#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:37:49 2020

@author: nora
"""
import re
from re import search
import pandas as pd
from pandas import DataFrame

df=pd.read_csv("data/Data_ALL_IF.csv")    

#funding_orgs = df["Funding Orgs"]

NewFunding=[]

for index, row in df.iterrows():
    

    funding_org = str(row['Funding Orgs'])
    
    #stripped = re.sub("[[].*[]]", "", funding_org)
    #NewFunding.append([stripped])

    if (re.search(r"(Fahd|KFUPM)", funding_org)):
        
        funding_class = "KFUPM"
        
    elif pd.isna(row['Funding Orgs']):
        funding_class = "Not funded"

    else:
        funding_class = "External"
        
    
    NewFunding.append([funding_class])
    
   

df2 = DataFrame(NewFunding, columns=['Funding Class'])
df2.to_csv (r'funding_class.csv', index = None, header=True)

