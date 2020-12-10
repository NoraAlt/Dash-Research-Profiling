# -*- coding: utf-8 -*-
"""
Project: University Research-Subject Profiling

Authors:
- Atheer Algherairy
- Nora Alturayeif
- Sarah Alyami
- Wadha Almatar

"""
# Import Libraries
import pathlib
import re
import json
from datetime import datetime
import flask
import dash
import dash_table
import matplotlib.colors as mcolors
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from precomputing import add_stopwords
from dash.dependencies import Output, Input, State
from dateutil import relativedelta
from wordcloud import WordCloud, STOPWORDS
from sklearn.manifold import TSNE
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import dash_table
import base64
import dash_html_components as html

nlp=en_core_web_sm.load()

"""
Users can select to run the dashboard with the whole dataset or a smaller
subset which then is evenly and consistently sampled accordingly.

It is worth mentioning that there is a time frame selection slider which
allows the user to look at specific time windows if there is desire to do so.

Once a data sample has been selected the user can select a department to look into
by using the dropdown or by clicking one of the bars on the right of departments
listed by number of publications.

Once a department has been selected, user can slecet from 5 taps to do deeper
inspections into interesting information about the publications to this specific
department:
    Tap1: Treemap of top subjects and subjects
    Tap2: A histogram with the most commonly used keywords in publications
    Tap3: Publication's impact factor divided into (high, average, low) based on the subject
    Tap4: Funding projects disturbuited on subjects
    Tap5: A wordcloud of most frequent words presented in the abstracts.


Note: All graphs and charts related to a selected department changed according to the
selected subset of dataset and time windows.


"""



# Read the DSR dataset
# The dataset contains publication from authors in KFUPM ranging from 2018 to 2020.
dataset_df = pd.read_csv("data/Data_ALL_IF.csv", index_col=0)
DATA_PATH = pathlib.Path(__file__).parent.resolve()
FILENAME = "data/Data_ALL_IF.csv"
FILENAME_PRECOMPUTED = "data/precomputed.json"
GLOBAL_DF = pd.read_csv(DATA_PATH.joinpath(FILENAME), header=0)

# Style settings
EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
PLOTLY_LOGO = "http://www.kfupm.edu.sa/Main_siteassets/images/kfupm_logo_ar.png"


'''
with open(DATA_PATH.joinpath(FILENAME_PRECOMPUTED)) as precomputed_file:
    PRECOMPUTED_LDA = json.load(precomputed_file)
'''



# casting the Publication Year column to datetime to make it easier in the rest of the code.
GLOBAL_DF["Publication Year"] = pd.to_datetime(
    GLOBAL_DF["Publication Year"], format="%m/%d/%Y"
)

test_png = 'data/heatmap_titles_S.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')


# Set additional stopwords in order to make the graphs more useful

ADDITIONAL_STOPWORDS = [
    "using",
    "used",
    "use",
    "method",
    "study",
    "based",
    "different",
    "propose",
    "proposed",
    "research",
    "paper",
    "approach",
    "System",
    "system",
    "model",

]

for stopword in ADDITIONAL_STOPWORDS:
    STOPWORDS.add(stopword)


# =============================================================================

# Function returns a subset of the provided dataframe.
# The sampling is evenly distributed and reproducible

def sample_data(dataframe, float_percent):

    print("making a local_df data sample with float_percent: %s" % (float_percent))
    return dataframe.sample(frac=float_percent, random_state=1)

# =============================================================================


# Function to get publication counts for unique department
def get_publication_count_by_department(dataframe):

    units_df=pd.read_csv(DATA_PATH.joinpath("data/KFUPM-Units.csv"), header=0)
    available_departments=units_df["Department"].str.strip()


    for index, row in units_df.iterrows():
        dep=row['Department'].strip()
        count = len(dataframe[ (dataframe['author1 Dep']== dep) | (dataframe['author2 Dep']==dep)| (dataframe['author3 Dep']== dep) | (dataframe['author4 Dep']==dep) | (dataframe['author5 Dep']== dep) | (dataframe['author6 Dep']==dep)])
        units_df.loc[index, 'Count'] = count

    # End For==================


    units_df.sort_values(by=['Count'], inplace=True, ascending=False)


    units_df.to_csv (r'data/KFUPM-Units.csv', index = None, header=True)



    values = available_departments.tolist()

    counts=units_df['Count'].tolist()
    return values, counts

def get_all_subjects(dataframe):
    # Get all subjects and sub-subjects
    all_subjects=dataframe['Subject1'].append(dataframe['Subject2'].append(dataframe['Subject3'].append(dataframe['Subject4'].append(dataframe['Subject5'].append(dataframe['Subject6']))))).str.strip().dropna().unique()
   

    return all_subjects

############## For NER (Named Entity Recognetion) table ######################


def get_funded_orgs_for_subject(dataframe, subject):
    temp_df=(dataframe[ (dataframe['Subject1']== subject) | (dataframe['Subject2']==subject)| (dataframe['Subject3']== subject) | (dataframe['Subject4']==subject) | (dataframe['Subject5']== subject) | (dataframe['Subject6']==subject)])


    #=====Preprocessing =====
    

    for index, row in temp_df.iterrows():
        funding_org = str(row['Funding Orgs'])
        if (re.search(r"(Fahd|Fand|fahd|fand|KFUPM)", funding_org)):
            temp_df.loc[index,'Funding Orgs'] = "King Fahd University of Petroleum & Minerals"
            
            
    #========================
    all_orgs=[]
    text= temp_df['Funding Orgs'].str.cat(sep=',')
    article=nlp(text)
    for ent in article.ents:
        if ent.label_ in ["ORG"]:
            all_orgs.append(ent.text)  
            

    #remove duplicates from list
    all_orgs = list(dict.fromkeys(all_orgs))
 
 
    
    fig = go.Figure(data=[go.Table(header=dict(values=['<b>Funding Organization</b>'],line_color='white', fill_color='steelblue', font=dict(color='white', size=15)),
                 cells=dict(values=[all_orgs], font = dict(color = 'darkslategray', size = 13), height=30, line_color='white', fill_color='whitesmoke', align='left'))
                     ])
    
    return fig
    
# =============================================================================

def calculate_sample_data(dataframe, sample_size, time_values, factor):
    print(
        "making sample_data with sample_size count: %s and time_values: %s"
        % (sample_size, time_values)
    )
    if time_values is not None:
        min_date = time_values[0]
        max_date = time_values[1]
        dataframe = dataframe[
            (dataframe["Publication Year"] >= min_date)
            & (dataframe["Publication Year"] <= max_date)
        ]

    # ====== Read units data ======
    units_df=pd.read_csv(DATA_PATH.joinpath("data/KFUPM-Units.csv"), header=0)


    # ====== Show results of all departments based on selected factor: ======
    # ----------------------------------------------------

    # 1- based on number of citaitions

    if  factor=='Number of Citation':

     for index, row in units_df.iterrows():
        dep=row['Department']
        total=0
        temp_df= dataframe[ (dataframe['author1 Dep']== dep) | (dataframe['author2 Dep']==dep)| (dataframe['author3 Dep']== dep) | (dataframe['author4 Dep']==dep) | (dataframe['author5 Dep']== dep) | (dataframe['author6 Dep']==dep)]
        total=temp_df['Times Cited, All Databases'].sum()
        units_df.loc[index, 'Total Citation'] = total

     units_df.sort_values(by=['Total Citation'], inplace=True, ascending=False)
     counts_sample=units_df['Total Citation'].tolist()
     values_sample=units_df["Department"][:sample_size].tolist()

     data = [
            {
                "x": values_sample,
                "y": counts_sample,
                "text": values_sample,
                "textposition": "auto",
                "type": "bar",
                "name": "",
                }
            ]
     layout = {
            "autosize": False,
            "margin": dict(t=10, b=10, l=40, r=0, pad=4),
            "xaxis": {"showticklabels": False},
            }
    # ====================================================

    # 2- based on impact factor of journals:

    elif  factor=='Impact Factor':

        for index, row in units_df.iterrows():
          dep=row['Department'] .strip()
          #count_high, percent_high=0
          #count_low, percent_low=0
          #count_avg, percent_avg=0
          temp_df=(dataframe[ (dataframe['author1 Dep']== dep) | (dataframe['author2 Dep']==dep)| (dataframe['author3 Dep']== dep) | (dataframe['author4 Dep']==dep) | (dataframe['author5 Dep']== dep) | (dataframe['author6 Dep']==dep)])
          count_high=temp_df['High'].sum()
          count_avg=temp_df['Avg'].sum()
          count_low=temp_df['Low'].sum()

          percent_high=(count_high/(count_high+count_avg+count_low)) *100
          percent_avg=(count_avg/(count_high+count_avg+count_low)) *100
          percent_low=(count_low/(count_high+count_avg+count_low)) *100

          units_df.loc[index,'High']=count_high
          units_df.loc[index,'Avg']=count_avg
          units_df.loc[index,'Low']=count_low
          units_df.loc[index,'Percentage High']=format(percent_high, ".2f")
          units_df.loc[index,'Percentage Avg']= format(percent_avg, ".2f")
          units_df.loc[index,'Percentage Low']= format(percent_low, ".2f")


        #units_df.sort_values(by=['Percentage High'], inplace=True, ascending=False)

        units_df.to_csv (r'data/KFUPM-Units.csv', index = None, header=True)

        trace1 = go.Bar(x=units_df['Department'], y=units_df['Percentage Low'], name='Low' )
        trace2 = go.Bar(x=units_df['Department'], y=units_df['Percentage Avg'], name='Avg')
        trace3 = go.Bar(x=units_df['Department'], y=units_df['Percentage High'], name='High')


        data=[trace1, trace2, trace3]
        layout = go.Layout(
            barmode="stack",
            margin= dict(t=10, b=230, l=40, r=0, pad=4),

        xaxis=dict(

            titlefont=dict(
                family='Courier New, monospace',
                size=12,
                color='#7f7f7f',
            )
        ),
        yaxis=dict(

            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
    # ==========================================================

    # 3- based on research funding (KFUPM, external, no funded):

    elif  factor=='Research Funding':

        for index, row in units_df.iterrows():
          dep=row['Department'] .strip()


          temp_df=(dataframe[ (dataframe['author1 Dep']== dep) | (dataframe['author2 Dep']==dep)| (dataframe['author3 Dep']== dep) | (dataframe['author4 Dep']==dep) | (dataframe['author5 Dep']== dep) | (dataframe['author6 Dep']==dep)])

          count_KFUPM = temp_df['Funding Class'].str.count('KFUPM').sum()
          count_External = temp_df['Funding Class'].str.count('External').sum()
          count_NA = temp_df['Funding Class'].str.count('Not funded').sum()
          count_all = len(temp_df.index) #count all rows

          percent_KFUPM = (count_KFUPM / count_all) *100
          percent_External = (count_External / count_all) *100
          percent_NA = (count_NA / count_all) *100

          units_df.loc[index,'KFUPM']= count_KFUPM
          units_df.loc[index,'External']= count_External
          units_df.loc[index,'NA']= count_NA
          units_df.loc[index,'Percentage KFUPM']=format(percent_KFUPM, ".2f")
          units_df.loc[index,'Percentage External']= format(percent_External, ".2f")
          units_df.loc[index,'Percentage NA']= format(percent_NA, ".2f")


        #units_df.sort_values(by=['Percentage KFUPM'], inplace=True, ascending=False)

        units_df.to_csv (r'data/KFUPM-Units.csv', index = None, header=True)

        trace1 = go.Bar(x=units_df['Department'], y=units_df['Percentage NA'], name='Not funded' )
        trace2 = go.Bar(x=units_df['Department'], y=units_df['Percentage External'], name='External')
        trace3 = go.Bar(x=units_df['Department'], y=units_df['Percentage KFUPM'], name='KFUPM')


        data=[trace1, trace2, trace3]
        layout = go.Layout(
            barmode="stack",
            margin= dict(t=10, b=230, l=40, r=0, pad=4),

        xaxis=dict(

            titlefont=dict(
                family='Courier New, monospace',
                size=12,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(

            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )


    # ====================================================
    
    # 3- based on number of publications (default option):

    else:
        for index, row in units_df.iterrows():
            dep=row['Department'] .strip()
            count = len(dataframe[ (dataframe['author1 Dep']== dep) | (dataframe['author2 Dep']==dep)| (dataframe['author3 Dep']== dep) | (dataframe['author4 Dep']==dep) | (dataframe['author5 Dep']== dep) | (dataframe['author6 Dep']==dep)])
            units_df.loc[index, 'Count'] = count

        units_df.sort_values(by=['Count'], inplace=True, ascending=False)
        counts_sample = units_df['Count'][:sample_size].tolist()
        values_sample=units_df["Department"][:sample_size].tolist()
        data = [
            {
                "x": values_sample,
                "y": counts_sample,
                "text": values_sample,
                "textposition": "auto",
                "type": "bar",
                "name": "",
                }
            ]
        layout = {
            "autosize": False,
            "margin": dict(t=10, b=10, l=40, r=0, pad=4),
            "xaxis": {"showticklabels": False},
            }

      # ==============================


    units_df.to_csv (r'data/KFUPM-Units.csv', index = None, header=True)


    return data, layout

#==============================================================================

# Function returns a subset of the provided dataframe of selected department.
# The sampling is evenly distributed and reproducible

def make_local_df(selected_dept, time_values, n_selection):

    print("redrawing wordcloud...")
    n_float = float(n_selection / 100)
    print("got time window:", str(time_values))
    print("got n_selection:", str(n_selection), str(n_float))
    print("got n_selection:", str(n_selection))


    local_df = sample_data(GLOBAL_DF, n_float)
    if time_values is not None:
        time_values = time_slider_to_date(time_values)
        local_df = local_df[
            (local_df["Publication Year"] >= time_values[0])
            & (local_df["Publication Year"] <= time_values[1])
        ]
    if selected_dept:
        dep=selected_dept
        local_df=local_df[ (local_df['author1 Dep']== dep) | (local_df['author2 Dep']==dep)| (local_df['author3 Dep']== dep) | (local_df['author4 Dep']==dep) | (local_df['author5 Dep']== dep) | (local_df['author6 Dep']==dep)]

        add_stopwords(selected_dept)


    return local_df

#==============================================================================

def make_marks_time_slider(mini, maxi):
    """
    A helper function to generate a dictionary that should look something like:
    {1420066800: '2015', 1427839200: 'Q2', 1435701600: 'Q3', 1443650400: 'Q4',
    1451602800: '2016', 1459461600: 'Q2', 1467324000: 'Q3', 1475272800: 'Q4',
     1483225200: '2017', 1490997600: 'Q2', 1498860000: 'Q3', 1506808800: 'Q4'}
    """
    step = relativedelta.relativedelta(months=+1)
    start = datetime(year=mini.year, month=1, day=1)
    end = datetime(year=maxi.year, month=maxi.month, day=30)
    ret = {}

    current = start
    while current <= end:
        current_str = int(current.timestamp())
        if current.month == 1:
            ret[current_str] = {
                "label": str(current.year),
                "style": {"font-weight": "bold"},
            }
        elif current.month == 4:
            ret[current_str] = {
                "label": "Q2",
                "style": {"font-weight": "lighter", "font-size": 7},
            }
        elif current.month == 7:
            ret[current_str] = {
                "label": "Q3",
                "style": {"font-weight": "lighter", "font-size": 7},
            }
        elif current.month == 10:
            ret[current_str] = {
                "label": "Q4",
                "style": {"font-weight": "lighter", "font-size": 7},
            }
        else:
            pass
        current += step
    # print(ret)
    return ret

#==============================================================================

def time_slider_to_date(time_values):
    """ TODO """
    min_date = datetime.fromtimestamp(time_values[0]).strftime("%c")
    max_date = datetime.fromtimestamp(time_values[1]).strftime("%c")
    print("Converted time_values: ")
    print("\tmin_date:", time_values[0], "to: ", min_date)
    print("\tmax_date:", time_values[1], "to: ", max_date)
    return [min_date, max_date]

#==============================================================================

# Function to generate the data format the dropdown dash component wants

def make_options_drop(values):

    ret = []
    for value in values:
        ret.append({"label": value, "value": value})
    return ret


#==============================================================================

# ====== Show results of the selected department based on clicked tap: ======
# This function returns figure data for 5 equally plots:
# subjects, frequent words, impact factor, funding, wordcloud
# Treemap of subjects, kewords frequency histogram, pie chart of funding projects,


def plotly_department(data_frame):

    keywords_text = list(data_frame["Author Keywords"].dropna().str.lower().values)

    # Get all subjects and sub-subjects
    all_subjects=data_frame['Subject1'].append(data_frame['Subject2'].append(data_frame['Subject3'].append(data_frame['Subject4'].append(data_frame['Subject5'].append(data_frame['Subject6']))))).str.strip().unique()


    #Extract publications impact factor and distuirbuit them on subject areas
    n_by_subject=[]
    #fund_by_subject=[]
    #--------
    external_fund_by_subj=[]
    notFunded_fund_by_subj=[]
    KFUPM_fund_by_subj=[]
    for i in all_subjects:
        temp_subject_df=data_frame[ (data_frame['Subject1'].str.strip()== i) | (data_frame['Subject2'].str.strip()== i)|(data_frame['Subject3'].str.strip()== i)|(data_frame['Subject4'].str.strip()== i)|(data_frame['Subject5'].str.strip()== i)|(data_frame['Subject6'].str.strip()== i)]
        n_by_subject.append(len(temp_subject_df))
        #count_funded = temp_subject_df['Funding Class'].str.count('KFUPM|External').sum()
        #fund_by_subject.append(count_funded)

        external_fund_by_subj.append(temp_subject_df['Funding Class'].str.count('External').sum())
        KFUPM_fund_by_subj.append(temp_subject_df['Funding Class'].str.count('KFUPM').sum())
        notFunded_fund_by_subj.append(temp_subject_df['Funding Class'].str.count('Not funded').sum())


####### Most frequent keywords

    if len(keywords_text) < 1:
        return {}, {}, {} , {}, {}

    # join all documents in corpus
    text = " ".join(list(keywords_text))

    word_cloud = WordCloud(stopwords=set(STOPWORDS), max_words=100, max_font_size=90)
    word_cloud.generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in word_cloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x_arr = []
    y_arr = []
    for i in position_list:
        x_arr.append(i[0])
        y_arr.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 80)

    trace = go.Scatter(
        x=x_arr,
        y=y_arr,
        textfont=dict(size=new_freq_list, color=color_list),
        hoverinfo="text",
        textposition="top center",
        hovertext=["{0} - {1}".format(w, f) for w, f in zip(word_list, freq_list)],
        mode="text",
        text=word_list,
    )

    layout = go.Layout(
        {
            "xaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 250],
            },
            "yaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 450],
            },
            "margin": dict(t=20, b=20, l=10, r=10, pad=4),
            "hovermode": "closest",
        }
    )

    wordcloud_figure_data = {"data": [trace], "layout": layout}
    word_list_top = word_list[:25]
    word_list_top.reverse()
    freq_list_top = freq_list[:25]
    freq_list_top.reverse()

    frequency_figure_data = {
        "data": [
            {
                "y": word_list_top,
                "x": freq_list_top,
                "type": "bar",
                "name": "",
                "orientation": "h",
            }
        ],
        "layout": {"height": "550", "margin": dict(t=20, b=20, l=100, r=20, pad=4)},
    }



######## Figure: stacked bar of publications impact factor distuirbuited on subject areas

    labels = ['High','Average','Low']
    count_high=data_frame['High'].sum()
    count_avg=data_frame['Avg'].sum()
    count_low=data_frame['Low'].sum()

    percent_high=format((count_high/(count_high+count_avg+count_low)) *100, ".2f")
    percent_avg=format((count_avg/(count_high+count_avg+count_low)) *100, ".2f")
    percent_low=format((count_low/(count_high+count_avg+count_low)) *100, ".2f")

    values = [percent_high, percent_avg, percent_low]

    impact_per_dep = go.Figure(data=[go.Pie(labels=labels, values=values)])
    #colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    colors= ['steelblue', 'powderblue', 'mintcream']
    impact_per_dep.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#111111', width=1)))




####### Figure: Pie chart of funding projects

    funding_data = []
    funding_data.append( go.Bar(
            x=all_subjects,
            y=notFunded_fund_by_subj,
            name="Not Funded"
        ))

    funding_data.append(
        go.Bar(
            x=all_subjects,
            y=external_fund_by_subj,
            name="External"
        ))

    funding_data.append (go.Bar(
            x=all_subjects,
            y=KFUPM_fund_by_subj,
            name="KFUPM"
        ))



    layout = go.Layout(barmode='stack', height= 700, margin=dict(t=50, b=50, l=5, r=5, pad=4))

    funding_figure_data = go.Figure(data=funding_data, layout=layout)



############## Figure: Treemap of subjects and subsubjects ###########

    subject_figure_data = go.Treemap(
        labels=all_subjects, parents=[""] * 15, values=n_by_subject
    )
    subject_figure_data_layout = go.Layout({"margin": dict(t=10, b=10, l=5, r=5, pad=4)})
    subject_figure_data_figure = {"data": [subject_figure_data], "layout": subject_figure_data_layout}
    return wordcloud_figure_data, frequency_figure_data,impact_per_dep,subject_figure_data_figure, funding_figure_data



"""
To clean up the code a bit, we decided to break it apart into sections.
For instance: LEFT_COLUMN is the input controls you see in that gray box on the
top left. The body variable is the overall structure which most other sections
go into. This just makes it ever so slightly easier to find the right spot to
add to or change without having to count too many brackets.
"""

################# Style settings for the dashboard ###########################

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("Deanship of Scientific Research: Scholarly Analysis | Research-Subject Profiling", className="ml-2")
                    ),
                ],
                align="left",
                no_gutters=True,
            ),
            href="http://www.kfupm.edu.sa/",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

LEFT_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="Select dataset size", className="display-5"),
        html.Hr(className="my-2"),

        html.Label("Filter by:", style={"marginTop": 50}, className="lead"),
        html.P(
            "(Departments is ordered by the selected factor)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),


        dcc.Dropdown(
            #id="option-drop",options=[{'label':'Number of Publication', 'value': 'Publication'}, {'label':'Number of Citation','value': 'Publication'}, {'label':'Impact Factor','value': 'Publication'}] ,clearable=False, style={"marginBottom": 50, "font-size": 12}
        id="option-drop", clearable=False, style={"marginBottom": 50, "font-size": 12}
        ),

        html.Label("Select time frame", className="lead"),
        html.Div(dcc.RangeSlider(id="time-window-slider"), style={"marginBottom": 50}),
        html.P(
            "(You can define the time frame down to month granularity)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        html.Label("Select percentage of dataset", className="lead"),
        html.P(
            "(Lower is faster. Higher is more precise)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Slider(
            id="n-selection-slider",
            min=1,
            max=100,
            step=1,
            marks={
                0: "0%",
                10: "",
                20: "20%",
                30: "",
                40: "40%",
                50: "",
                60: "60%",
                70: "",
                80: "80%",
                90: "",
                100: "100%",
            },
            value=100,
        ),

    ]
)


WORDCLOUD_PLOTS = [
    dbc.CardHeader(html.H5("Information per department/center")),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-data-alert",
        color="warning",
        style={"display": "none"},
    ),
    #
     html.Label("Select a department/center", style={"marginTop": 50}, className="lead"),
        html.P(
            "(You can use the dropdown or click the barchart on the top)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id=" dept-drop", clearable=False, style={"marginBottom": 50, "font-size": 12}
        ),
    dbc.CardBody(
        [
            dbc.Row(
                [

                    dbc.Col(
                        [
                            dcc.Tabs(
                                id="tabs",
                                children=[
                                    #################
                                    dcc.Tab(
                                        label="Subjects",
                                        children=[
                                            dcc.Loading(
                                                id="loading-Subjects",
                                                children=[
                                                    dcc.Graph(id="department_Subjects")
                                                ],
                                                type="default",
                                            )
                                        ],
                                    ),

                                    dcc.Tab(
                                        label="Frequent Keywords",
                                        children=[
                                             dcc.Loading(
                                                 id="loading-frequencies",
                                                 children=[dcc.Graph(id="frequency_figure")],
                                                 type="default",
                                                 )
                                             ],
                    ),
                                    #################
                                    dcc.Tab(
                                        label="Impact Factor",
                                        children=[
                                            dcc.Loading(
                                                id="loading-treemap",
                                                children=[dcc.Graph(id=" dept-treemap")],
                                                type="default",
                                            )
                                        ],
                                    ),
                                    #################
                                    dcc.Tab(
                                        label="Funding per Subject",
                                        children=[
                                            dcc.Loading(
                                                id="loading-funding",
                                                children=[dcc.Graph(id="department-funding")],
                                                type="default",
                                            )
                                        ],
                                    ),

                                    dcc.Tab(
                                        label="Wordcloud",
                                        children=[
                                            dcc.Loading(
                                                id="loading-wordcloud",
                                                children=[
                                                    dcc.Graph(id=" dept-wordcloud")
                                                ],
                                                type="default",
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                        md=12,
                    ),
                ]
            )
        ]
    )
]


TOP_DEPTS_PLOT = [
    dbc.CardHeader(html.H5("Departments ordered based on selected filter")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading- depts-hist",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert- dept",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id=" dept-sample", style={'height':'80vh'}),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]


FUNDING_PLOTS = [
    dbc.CardHeader(html.H5("Funding Organization")),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-data-alert1",
        color="warning",
        style={"display": "none"},
    ),
    #
     html.Label("Select a subject", style={"marginTop": 50}, className="lead"),
        html.P(
            "(You can use the dropdown or click the barchart on the top)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="subjects-drop", clearable=False, style={"marginBottom": 50, "font-size": 12}
        ),
        dcc.Graph(id="funding_orgs", style={'height':'80vh'})
        #dash_table.DataTable(id='funding_orgs')
  
]

SIMILARITY_PLOT= [
    dbc.CardHeader(html.H5("SIMILARITY HEATMAP")),
    html.Img(src='data:image/png;base64,{}'.format(test_base64)),
    
    ]


BODY = dbc.Container(
    [
        #dbc.Row([dbc.Col(dbc.Card(TOP_BIGRAM_COMPS)),], style={"marginTop": 30}),

        #dbc.Row([dbc.Col(dbc.Card(TOP_BIGRAM_PLOT)),], style={"marginTop": 30}),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=3, align="center"),
                dbc.Col(dbc.Card(TOP_DEPTS_PLOT), md=9),
            ],
            style={"marginTop": 30},
        ),


        
        dbc.Row([dbc.Col([dbc.Card(WORDCLOUD_PLOTS)])], style={"marginTop": 50}),
        dbc.Card(FUNDING_PLOTS),
        dbc.Row([dbc.Col([dbc.Card(SIMILARITY_PLOT)])], style={"marginTop": 50}),

    ],
    className="mt-12",
)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # for Heroku deployment

app.layout = html.Div(children=[NAVBAR, BODY])

"""
#  Callbacks
"""

@app.callback(
    [
        Output("time-window-slider", "marks"),
        Output("time-window-slider", "min"),
        Output("time-window-slider", "max"),
        Output("time-window-slider", "step"),
        Output("time-window-slider", "value"),
    ],
    [Input("n-selection-slider", "value")],
)
def populate_time_slider(value):
    """
    Depending on our dataset, we need to populate the time-slider
    with different ranges. This function does that and returns the
    needed data to the time-window-slider.
    """
    value += 0
    min_date = GLOBAL_DF["Publication Year"].min()
    max_date = GLOBAL_DF["Publication Year"].max()

    marks = make_marks_time_slider(min_date, max_date)
    min_epoch = list(marks.keys())[0]
    max_epoch = list(marks.keys())[-1]

    return (
        marks,
        min_epoch,
        max_epoch,
        (max_epoch - min_epoch) / (len(list(marks.keys())) * 3),
        [min_epoch, max_epoch],
    )


@app.callback(
    Output(" dept-drop", "options"),
    [Input("time-window-slider", "value"), Input("n-selection-slider", "value")],
)
def populate_dept_dropdown(time_values, n_value):
    """ TODO """
    print(" dept-drop: TODO USE THE TIME VALUES AND N-SLIDER TO LIMIT THE DATASET")
    if time_values is not None:
        pass
    n_value += 1
    dept_names, counts = get_publication_count_by_department(GLOBAL_DF)
    counts.append(1)


    return make_options_drop( dept_names)

@app.callback(
    Output("subjects-drop", "options"),
    [Input("time-window-slider", "value"), Input("n-selection-slider", "value"), Input(" dept-drop", "value")],
)
def populate_subject_dropdown(time_values, n_value, dep):
    """ TODO """
    print(" subjects-drop: TODO USE THE TIME VALUES AND N-SLIDER TO LIMIT THE DATASET")



    #===========Filtered (subjects in selected dep)=====
    local_filtered_by_dep=make_local_df(dep, time_values,n_value)

    subjects = get_all_subjects(local_filtered_by_dep)
    
    #============NOt filtered (ALL subjects)===

    #subjects = get_all_subjects(GLOBAL_DF)
    #============
   
    ret = []
    for item in subjects:
        if item is not None:
           ret.append({"label":item, "value": item})

    return (ret)

@app.callback(
    Output("option-drop", "options"),
    [Input("time-window-slider", "value"), Input("n-selection-slider", "value")],
)
def populate_options_dropdown(time_values, n_value):
    """ TODO """
    print(" dept-drop: TODO USE THE TIME VALUES AND N-SLIDER TO LIMIT THE DATASET")
    if time_values is not None:
        pass
    n_value += 1
    options = ["Number of Publications", "Number of Citation", "Impact Factor", "Research Funding"]
    #options=[{'label':'Number of Publications', 'value': 'Number of Publications'}, {'label':'Number of Citation','value': 'Number of Citation'}, {'label':'Impact Factor','value': 'Impact Factor'}]

    return make_options_drop(options)


@app.callback(
    [Output(" dept-sample", "figure"), Output("no-data-alert- dept", "style")],
    [Input("option-drop", "value"), Input("n-selection-slider", "value"), Input("time-window-slider", "value")],
)
def update_dept_sample_plot(factor, n_value, time_values):
    """ TODO """
    print("redrawing  dept-sample...")
    print("\tn is:", n_value)
    print("\ttime_values is:", time_values)
    if time_values is None:
        return [{}, {"display": "block"}]
    n_float = float(n_value / 100)
    dept_sample_count = 39
    local_df = sample_data(GLOBAL_DF, n_float)
    min_date, max_date = time_slider_to_date(time_values)
    data, layout = calculate_sample_data(
    local_df,  dept_sample_count, [min_date, max_date], factor
   )


    print("redrawing  dept-sample...done")
    return [{"data": data, "layout": layout}, {"display": "none"}]


@app.callback(
    [


        Output(" dept-wordcloud", "figure"),
        Output("frequency_figure", "figure"),
        Output(" dept-treemap", "figure"),
        Output("no-data-alert", "style"),


        Output("department_Subjects", "figure"),
        Output("department-funding", "figure"),

    ],
    [
        Input(" dept-drop", "value"),
        Input("time-window-slider", "value"),
        Input("n-selection-slider", "value"),
    ],
)
def update_wordcloud_plot(value_drop, time_values, n_selection):
    """ Callback to rerender wordcloud plot """
    local_df = make_local_df(value_drop, time_values, n_selection)
    wordcloud, frequency_figure, treemap , subject, funding= plotly_department(local_df)
    alert_style = {"display": "none"}
    if (wordcloud == {}) or (frequency_figure == {}) or (treemap == {}) or (subject == {}) or (funding == {}):
        alert_style = {"display": "block"}
    print("redrawing  dept-wordcloud...done")
    return (wordcloud, frequency_figure, treemap, alert_style, subject, funding)


@app.callback(
    [
        Output("funding_orgs", "figure"),
        Output("no-data-alert1", "style"),

    ],
    [
        Input("subjects-drop", "value"),
        Input(" dept-drop", "value"),
        Input("time-window-slider", "value"),
        Input("n-selection-slider", "value"),
    ],
)

######### For NER (Named Entity Recognetion) table ###########################

def update_funding_subject_plot(sub,dep,time_values,n_value):


    local_filtered_by_dep=make_local_df(dep, time_values,n_value)

    #subjects = get_all_subjects(local_filtered_by_dep)
    fig=get_funded_orgs_for_subject(local_filtered_by_dep,sub)
    alert_style = {"display": "none"}



    print("redrawing funding orgs")
    return fig,alert_style


##############################################################################

@app.callback(Output(" dept-drop", "value"), [Input(" dept-sample", "clickData")])
def update_dept_drop_on_click(value):
    """ TODO """
    if value is not None:
        selected_dept = value["points"][0]["x"]
        return selected_dept
    return "Mechanical Engineering Department"

########################### Run Application ##################################

if __name__ == "__main__":
    app.run_server(debug=True)
