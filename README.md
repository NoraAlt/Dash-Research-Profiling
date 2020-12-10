

## About this Project
The general objective of this study is to perform exploratory analysis on a scholarly dataset from KFUPM, to create a research profile for each departments/centre. The research profile describes several information such as, the  subjects addressed by researchers, the impact of the publications, the degree of funding received and most importantly, how do these departments/centers compare in regards to their research activity.

The Dash framework released by plotly is used to visualize the results of our scholarly analysisk, which show how plotly's excellent dash framework can be used for Natural Language Processing (NLP).

## How to run this app

To run this app first clone repository and then open a terminal to the app folder.

```
git clone https://github.com/NoraAlt/Dash-Research-Profiling.git

```

Install the requirements:

```
pip install -r requirements.txt
```
Run the app:

```
python app.py
```
You can run the app on your browser at http://127.0.0.1:8050

## Files
```
app.py
```
App file to browse the dash-board.

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

Note: All graphs and charts related to a selected department changed according to the selected subset of dataset and time windows.

```
Preprocessing/departments.py
Preprocessing/Funding.py
Preprocessing/PublicationText_per_Dep.py
```
For Preprocessing and information extraction.

```
Similarity/normalization.py
Similarity/Similarity.py
Similarity/utils.py
```
Similarities between two departments/centers were measured based on a comparison of the titles of their publications by leveraging NLP to semantically compare texts.
- normalization.py : for data preprocessing.
- Similarity.py : to compute the cosine similarity between two texts vectors.
- utils.py : to plot a heat-map of the similarity scores to visually assess which two departments/centres are most similar and most dissimilar to each other
