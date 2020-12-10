#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:51:14 2020

@author: atheeralgherairy
"""


import pandas as pd


# Function to get publication counts for unique department   
def get_publication_count_by_department():
    
    units_df=pd.read_csv("/Users/atheeralgherairy/Desktop/Similarity/KFUPM-Units.csv", header=0)    
    dataframe=pd.read_csv("/Users/atheeralgherairy/Desktop/Similarity/Data_ALL_IF.csv", header=0)    

    
    available_departments=units_df["Department"].str.strip()
    

    for index, row in units_df.iterrows():
        dep=row['Department'].strip()
        
        temp_df=(dataframe[ (dataframe['author1 Dep']== dep) | (dataframe['author2 Dep']==dep)| (dataframe['author3 Dep']== dep) | (dataframe['author4 Dep']==dep) | (dataframe['author5 Dep']== dep) | (dataframe['author6 Dep']==dep)])  

      
        
        All_Abstracts=""
        for i,abs in temp_df['Abstract'].iteritems():
            All_Abstracts=All_Abstracts+"|| "+str(abs)
            
        All_titles=""
        for i,abs in temp_df['Article Title'].iteritems():
            All_titles=All_titles+"|| "+str(abs)
           
        All_keywords=""
        for i,abs in temp_df['Author Keywords'].iteritems():
            All_keywords=All_keywords+"|| "+str(abs)
            
        
        units_df.loc[index, 'All Abstracts'] = All_Abstracts
        units_df.loc[index, 'All titles'] = All_titles
        units_df.loc[index, 'All Keywords'] = All_keywords
       
    # End For==================



    
    #units_df.to_csv (r'/Users/atheeralgherairy/Desktop/Similarity/KFUPM-Units_Text.csv', index = None, header=True)
    units_df.to_excel(r'/Users/atheeralgherairy/Desktop/Similarity/KFUPM-Units_NEW.xlsx', index = False)


    

    departments = available_departments.tolist()

    abstracts=units_df['All Abstracts'].tolist()
    return departments, abstracts

def listToString(s):  
    
    # initialize an empty string 
    str1 = " " 
    
    # return string   
    return (str1.join(str(s))) 


get_publication_count_by_department()