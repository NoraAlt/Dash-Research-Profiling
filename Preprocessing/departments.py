#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:16:21 2020

@author: nora
"""

# import libraries

import re
from re import search

import pandas as pd
from pandas import DataFrame

import numpy as np
import os
import plotly.express as px
import matplotlib.pyplot as plt

pd.options.plotting.backend = 'plotly'


def check_Dictionary(text):
    for key, value in dictionary.items():
        if (re.search(key, text)):
            return value


# load data in df

df = pd.read_csv('Departments.csv')

dictionary = {
    r"Elect Engn": "Electrical Engineering Department",
    r"Mech Engn": "Mechanical Engineering Department",
    r"Petr Engn": "Petroleum Engineering Department",
    r"Comp Engn": "Computer Engineering Department",
    r"Chem Engn": "Chemical Engineering Department",
    r"(Chem Dept|Dept Chem)": "Chemistry Department",
    r"Commun & IT": "Center for Communication & IT",
    r"(Engn Res|Ctr Engn)": "Center for Engineering",
    r"Civil": "Civil & Environmental Engineering Department",
    r"English Language": "English Language Department",
    r"Global & Social Studies": "Global and Social Studies Department",
    r"Phys Educ": "Physical Education Department",
    r"(Preparatory|Prep)": "Physical Education Department",
    r"Informat & Comp Sci": "Information & Computer Science Department",
    r"Syst Engn": "Systems Engineering Department",
    r"Aerosp Engn": "Aerospace Engineering Department",
    r"Architectural Engn": "Architectural Engineering Department",
    r"Architectural": "Architectural Department",
    r"City & Reg Planning": "City & Regional Planning Department",
    r"Construct": "Construction Engineering & Management Department",
    r"Accounting & Finance": "Accounting & Finance Department",
    r"Management & Mkt": "Management & Marketing Department",
    r"Informat Syst & Operat Management": "Information System & Operations Management Department",
    r"Geo Sci": "Geosciences Department",
    r"Ctr Integrat Petr Res": "Center for Integrative Petroleum Research",
    r"Math & Stat": "Mathematics & Statistics Department",
    r"Life Sci": "Life Sciences Department",
    r"Phys": "Phys Department",
    r"Dammam Community Coll": "Dammam Community College",
    r"Commun & IT":"Center for Communication & IT",
    r"Refining & Petrochem":"Center for Refining & Petrochemicals",
    r"Environm & Water":"Center for Environment & Water",
    r"Corros":"Center of Research Excellence in Corrosion",
    r"Renewable Energy":"Center of Research Excellence in Renewable Energy",
    r"Energy Effciency":"Center of Research Excellence in Energy Efficiency",
    r"Nanotechnol":"Center of Research Excellence in Nanotechnology",
    r"Islamic Banking & Finance":"Center of Research Excellence in Islamic Banking and Finance",
    r"Accounting & Management Informat Syst":"Accounting & Management Information System"
}


NewDepartments=[]

for index, row in df.iterrows():


    # ===================AUTHOR 1=============================
    a_col1 = str(row['A1Uni'])
    a_col2 = str(row['A1Dep1'])
    a_col3 = str(row['A1Dep2'])
    a_col4 = str(row['A1Dep3'])

    if (re.search(r"(Fahd|KFUPM)", a_col1)):
        if check_Dictionary(a_col2):
            a1_NewDept=check_Dictionary(a_col2)
        elif check_Dictionary(a_col3):
            a1_NewDept=check_Dictionary(a_col3)
        elif check_Dictionary(a_col4):
            a1_NewDept=check_Dictionary(a_col4)
        else:
            a1_NewDept="UNKNOWN KFUPM Department"
    else:
        a1_NewDept="NA"

    # ===================AUTHOR 2=============================

    a_col1 = str(row['A2Uni'])
    a_col2 = str(row['A2Dep1'])
    a_col3 = str(row['A2Dep2'])
    a_col4 = str(row['A2Dep3'])

    if (re.search(r"(Fahd|KFUPM)", a_col1)):
        if check_Dictionary(a_col2):
            a2_NewDept=check_Dictionary(a_col2)
        elif check_Dictionary(a_col3):
            a2_NewDept=check_Dictionary(a_col3)
        elif check_Dictionary(a_col4):
            a2_NewDept=check_Dictionary(a_col4)
        else:
            a2_NewDept="UNKNOWN KFUPM Department"
    else:
        a2_NewDept="NA"
        
    
      # ===================AUTHOR 3=============================

    a_col1 = str(row['A3Uni'])
    a_col2 = str(row['A3Dep1'])
    a_col3 = str(row['A3Dep2'])
    a_col4 = str(row['A3Dep3'])


    if (re.search(r"(Fahd|KFUPM)", a_col1)):
        if check_Dictionary(a_col2):
            a3_NewDept=check_Dictionary(a_col2)
        elif check_Dictionary(a_col3):
            a3_NewDept=check_Dictionary(a_col3)
        elif check_Dictionary(a_col4):
            a3_NewDept=check_Dictionary(a_col4)
        else:
            a3_NewDept="UNKNOWN KFUPM Department"
    else:
        a3_NewDept="NA"
        
        
      # ===================AUTHOR 4=============================

    a_col1 = str(row['A4Uni'])
    a_col2 = str(row['A4Dep1'])
    a_col3 = str(row['A4Dep2'])
    a_col4 = str(row['A4Dep3'])
    
    if (re.search(r"(Fahd|KFUPM)", a_col1)):
        if check_Dictionary(a_col2):
            a4_NewDept=check_Dictionary(a_col2)
        elif check_Dictionary(a_col3):
            a4_NewDept=check_Dictionary(a_col3)
        elif check_Dictionary(a_col4):
            a4_NewDept=check_Dictionary(a_col4)
        else:
            a4_NewDept="UNKNOWN KFUPM Department"
    else:
        a4_NewDept="NA"  
        
      # ===================AUTHOR 5=============================

    a_col1 = str(row['A5Uni'])
    a_col2 = str(row['A5Dep1'])
    a_col3 = str(row['A5Dep2'])
    a_col4 = str(row['A5Dep3'])

    if (re.search(r"(Fahd|KFUPM)", a_col1)):
        if check_Dictionary(a_col2):
            a5_NewDept=check_Dictionary(a_col2)
        elif check_Dictionary(a_col3):
            a5_NewDept=check_Dictionary(a_col3)
        elif check_Dictionary(a_col4):
            a5_NewDept=check_Dictionary(a_col4)
        else:
            a5_NewDept="UNKNOWN KFUPM Department"
    else:
        a5_NewDept="NA"
        
      # ===================AUTHOR 6=============================

    a_col1 = str(row['A6Uni'])
    a_col2 = str(row['A6Dep1'])
    a_col3 = str(row['A6Dep2'])
    a_col4 = str(row['A6Dep3'])

    if (re.search(r"(Fahd|KFUPM)", a_col1)):
        if check_Dictionary(a_col2):
            a6_NewDept=check_Dictionary(a_col2)
        elif check_Dictionary(a_col3):
            a6_NewDept=check_Dictionary(a_col3)
        elif check_Dictionary(a_col4):
            a6_NewDept=check_Dictionary(a_col4)
        else:
            a6_NewDept="UNKNOWN KFUPM Department"
    else:
        a6_NewDept="NA"
        
     # ==========================================================
     
    NewDepartments.append([a1_NewDept,a2_NewDept,a3_NewDept,a4_NewDept,a5_NewDept,a6_NewDept])
    
# End For ==========================================================

df2 = DataFrame(NewDepartments, columns=['author1 Dep', 'author2 Dep','author3 Dep','author4 Dep','author5 Dep','author6 Dep'])
df2.to_csv (r'departments_V2.csv', index = None, header=True)

