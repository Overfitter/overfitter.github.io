---
layout: post
title: NLP-Hack : Restaurant Recommendation Engine Pipeline
subtitle: Recommend Top 3 Restaurant names based on User's Input Parameters (Content Based Recommender System)
# gh-repo: daattali/beautiful-jekyll
# gh-badge: [star, fork, follow]
tags: [nlp, word-embeddings, data-science, machine-learning]
# comments: true
---

<h1>Overview<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Problem-Statement-&amp;-Approach" data-toc-modified-id="Problem-Statement-&amp;-Approach-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Problem Statement &amp; Approach</a></span></li><li><span><a href="#Loading-Required-Modules" data-toc-modified-id="Loading-Required-Modules-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Loading Required Modules</a></span></li><li><span><a href="#Helper-Functions" data-toc-modified-id="Helper-Functions-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Helper Functions</a></span></li><li><span><a href="#Main-Functions" data-toc-modified-id="Main-Functions-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Main Functions</a></span></li><li><span><a href="#Directory-Setup" data-toc-modified-id="Directory-Setup-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Directory Setup</a></span></li><li><span><a href="#Data-Preparation-&amp;-Exploration" data-toc-modified-id="Data-Preparation-&amp;-Exploration-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Data Preparation &amp; Exploration</a></span><ul class="toc-item"><li><span><a href="#Load-Datasets" data-toc-modified-id="Load-Datasets-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Load Datasets</a></span></li><li><span><a href="#Missing-Value-Count" data-toc-modified-id="Missing-Value-Count-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Missing Value Count</a></span></li><li><span><a href="#Data-Cleaning-&amp;-Feature-Engineering" data-toc-modified-id="Data-Cleaning-&amp;-Feature-Engineering-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Data Cleaning &amp; Feature Engineering</a></span></li><li><span><a href="#Data-Exploration" data-toc-modified-id="Data-Exploration-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Data Exploration</a></span><ul class="toc-item"><li><span><a href="#Most-Popular-Cuisines-served-by-Restaurants" data-toc-modified-id="Most-Popular-Cuisines-served-by-Restaurants-6.4.1"><span class="toc-item-num">6.4.1&nbsp;&nbsp;</span>Most Popular Cuisines served by Restaurants</a></span></li><li><span><a href="#Most-Popular-Restaurant's-Locations" data-toc-modified-id="Most-Popular-Restaurant's-Locations-6.4.2"><span class="toc-item-num">6.4.2&nbsp;&nbsp;</span>Most Popular Restaurant's Locations</a></span></li><li><span><a href="#Most-Popular-Restaurant's-Type" data-toc-modified-id="Most-Popular-Restaurant's-Type-6.4.3"><span class="toc-item-num">6.4.3&nbsp;&nbsp;</span>Most Popular Restaurant's Type</a></span></li><li><span><a href="#Dishes-Liked-vs-Online-Order" data-toc-modified-id="Dishes-Liked-vs-Online-Order-6.4.4"><span class="toc-item-num">6.4.4&nbsp;&nbsp;</span>Dishes Liked vs Online Order</a></span></li><li><span><a href="#Cuisines-Liked-vs-Online-Order" data-toc-modified-id="Cuisines-Liked-vs-Online-Order-6.4.5"><span class="toc-item-num">6.4.5&nbsp;&nbsp;</span>Cuisines Liked vs Online Order</a></span></li><li><span><a href="#Location's-with-most-votes-and-best-ratings-for-their-Restaurants" data-toc-modified-id="Location's-with-most-votes-and-best-ratings-for-their-Restaurants-6.4.6"><span class="toc-item-num">6.4.6&nbsp;&nbsp;</span>Location's with most votes and best ratings for their Restaurants</a></span></li><li><span><a href="#Restaurant's-Type-with-most-votes-and-best-ratings" data-toc-modified-id="Restaurant's-Type-with-most-votes-and-best-ratings-6.4.7"><span class="toc-item-num">6.4.7&nbsp;&nbsp;</span>Restaurant's Type with most votes and best ratings</a></span></li><li><span><a href="#Book-Table-vs-Rating" data-toc-modified-id="Book-Table-vs-Rating-6.4.8"><span class="toc-item-num">6.4.8&nbsp;&nbsp;</span>Book Table vs Rating</a></span></li><li><span><a href="#Book-Table-vs-Votes" data-toc-modified-id="Book-Table-vs-Votes-6.4.9"><span class="toc-item-num">6.4.9&nbsp;&nbsp;</span>Book Table vs Votes</a></span></li><li><span><a href="#Book-Table-vs-Costs" data-toc-modified-id="Book-Table-vs-Costs-6.4.10"><span class="toc-item-num">6.4.10&nbsp;&nbsp;</span>Book Table vs Costs</a></span></li><li><span><a href="#Rating-&amp;-Votes-&amp;-Costs-Distribution" data-toc-modified-id="Rating-&amp;-Votes-&amp;-Costs-Distribution-6.4.11"><span class="toc-item-num">6.4.11&nbsp;&nbsp;</span>Rating &amp; Votes &amp; Costs Distribution</a></span></li><li><span><a href="#Most-Frequent-Words---In-Reviews-Text" data-toc-modified-id="Most-Frequent-Words---In-Reviews-Text-6.4.12"><span class="toc-item-num">6.4.12&nbsp;&nbsp;</span>Most Frequent Words - In Reviews Text</a></span></li><li><span><a href="#WordCloud-of-Restaurant's-Reviews" data-toc-modified-id="WordCloud-of-Restaurant's-Reviews-6.4.13"><span class="toc-item-num">6.4.13&nbsp;&nbsp;</span>WordCloud of Restaurant's Reviews</a></span></li></ul></li></ul></li><li><span><a href="#Recommendation-Engine" data-toc-modified-id="Recommendation-Engine-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Recommendation Engine</a></span><ul class="toc-item"><li><span><a href="#Data-Preparation-for-Recommendation-Engine" data-toc-modified-id="Data-Preparation-for-Recommendation-Engine-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Data Preparation for Recommendation Engine</a></span></li><li><span><a href="#User-Input-Parameters" data-toc-modified-id="User-Input-Parameters-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>User Input Parameters</a></span></li><li><span><a href="#Content-Based-Recommender-System" data-toc-modified-id="Content-Based-Recommender-System-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>Content Based Recommender System</a></span></li></ul></li></ul></div>


```python
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
```




<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>



### Problem Statement & Approach


**Problem 1:** Analyse the given data having user reviews and other information of restaurants. Perform an EDA to find out useful insights to improve overall restaurant experience in Bengaluru.

- Every insight is supported and proven by Data Visualizations
- Create a brief EDA report, which helps us understand the nature of given dataset for the given problem



**Problem 2:** Build a recommendation system, where user should be able to express what he/she wants and system should recommend relevant restaurants, to her/his liking.

Recommend Top 3 Restaurant names based on these 4 input parameters:
- Location
- Budget
- Cuisine = e.g: Continental/Indian/Japanese etc
- User’s descriptive ordering information, as free text input. (e.g: good ambiance restaurant, serving fish)


The outline of this notebook is as follows:

1. **Exploratory Data Analysis (EDA)** - Analyzing the given data having user reviews and other information of restaurants. 

Performed Following Data Preparation & Analysis steps on restaurant's dataset:

- Checked the missing value counts for given variables
- Data Cleaning/Feature Engineering
    - Calculated # Reviews/Reviews/Ratings from reviews_list
    - Cleaned menu_items and calculated of # menu items
    - Cleaned the reviews using gensim text cleaning module
- Most Popular Cuisines served by Restaurant's
- Most Popular Restaurant's Locations
- Most Popular Restaurant's Type
- Dishes Liked in Online Order
- Cuisines Liked in Online Order
- Location's with most votes and best ratings from their Restaurant's
- Restaurant's Type with most votes and best ratings
- Book Table vs Rating/Votes/Costs
- Rating & Votes & Costs Correlation
- Most Frequent Words in Reviews Text
- WordCloud of Restaurant's Reviews
  
  
2. **Recommendation Engine** - Recommend Top 3 Restaurant names based on given input parameters using Content Based Recommender System


- Computed the similarity between restaurants based on certain parameter/metric(s) and suggests restaurants that are most similar to a particular restaurant that a user liked (user input).

- As a first step, to reduce the search space, we will take user inputs (Location, Cuisines, Budget) and filter the dataset. After applying these filteres we will check the cosine similarity between User Description and Restaurant Review (Using TF-IDF vectorizer). Then, we will come up with top 3 restaurant's based on User's Input Parameter's


3. **Appendix** - Leveraged following modules for data analysis and recommender system


- numpy/pandas : For Data Loading/Pre-Processing
- os: For directory setups
- gensim/nltk/re: For Text Cleaning/Processing
- plotly/wordclouds: For Visualization
- Scikit - Learn: To create TF-IDF vectorizer, matrix and cosine similarity
- tqdm: To check the progress of a functions/process

### Loading Required Modules


```python
# =============================================================================
# General ML Utitlies
# =============================================================================
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import gc # Garbage collector
import scipy # For sparse matrix
import os 
import pickle
import re
import time
import math
import ast
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import matplotlib.style as style
style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# NLTK Utitlies
# =============================================================================
import nltk
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
from nltk.tokenize import WordPunctTokenizer, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}


# =============================================================================
# Scikit-Learn Utitlies
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# =============================================================================
# Gensim Utitlies
# =============================================================================
from gensim import utils
import gensim.parsing.preprocessing as gsp


# =============================================================================
# Plotly Utitlies
# =============================================================================
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

# WordCloud
from wordcloud import WordCloud, STOPWORDS

# =============================================================================
# Tqdm utlities
# =============================================================================
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm
tqdm.pandas(desc='Progress')


# =============================================================================
# Other Utilities
# =============================================================================

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

pd.set_option('display.max_colwidth', -1)
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>



### Helper Functions


```python
### General Functions ###

def mkdir(*args):
    """
    To create directory for a given path
    """
    path = os.path.join(*args)
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    return path

def neat_print_sperator(name):
    """
    A function to create sperations between displays and prints
    """
    spacing_len = 95 - len(name)
    print('{} {} {}'.format('_'*math.floor((spacing_len/2)), name, '_'*math.ceil(spacing_len/2)))
    return
```


```python
### Dataframe related functions ###

def load_dataframe(file_path, file_type = "csv"):
    """
    A function to load datasets in csv/xlsx format
    """
    if file_type=="csv":
        df = pd.read_csv(file_path, index_col = None, encoding = "ISO-8859-1")
    elif file_type=="xlsx":
        df = pd.read_excel(file_path, index_col = None, encoding = "ISO-8859-1")
    return df

def save_dataframe(data_frame, file_path, file_type = "csv"):
    if file_type=="csv":
        data_frame.to_csv(file_path, index = False)
    elif file_type=="xlsx":
        data_frame.to_xlsx(file_path, index = False)

def gradient_dataframe(df, caption):
    """
    Converting dataframe into Gradient DataFrame
    """
    
    agg_df = df.style.set_table_styles(
            [{'selector': 'tr:nth-of-type(odd)',
              'props': [('background', '#eee'),
                       ('text-align', 'left')]}, 
             {'selector': 'tr:nth-of-type(even)',
              'props': [('background', 'white'), ('text-align', 'left')]},
             {'selector': 'th',
              'props': [('background', '#808080'), 
                        ('color', 'white'),
                        ('font-family', 'verdana'),
                       ('text-align', 'left')]},
             {'selector': 'td',
              'props': [('font-family', 'verdana'), ('text-align', 'left')]},
            ]
            ).hide_index().format(formatter=None).set_caption(f'{caption}')\
                .background_gradient(cmap='Blues')
    return agg_df

def missing_value_of_data(data):
    """
    Function to find out missing values in the dataset
    """
    total=data.isnull().sum().sort_values(ascending=False)
    percentage=np.round(total/data.shape[0]*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
```


```python
### Plotly related functions ###

def plotly_bar_pie_chart(cnt_srs, layout_title, pie_title):
    """
    Plotly function to Plot Bar & Pie Charts
    """

    trace = go.Bar(
        x=cnt_srs.index,
        y=cnt_srs.values,
        marker=dict(
            color=cnt_srs.values,
            colorscale = 'Picnic',
            reversescale = True
        ),
    )

    layout = go.Layout(
        title=layout_title,
        font=dict(size=12)
    )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    py.iplot(fig, filename=pie_title)

    ## target distribution ##
    labels = (np.array(cnt_srs.index))
    sizes = (np.array((cnt_srs / cnt_srs.sum())*100))

    trace = go.Pie(labels=labels, values=sizes)
    layout = go.Layout(
        title=pie_title,
        font=dict(size=12)
    )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    py.iplot(fig, filename="usertype")

def horizontal_bar_chart(df, color):
    """
    Custom function for Horizontal Bar Chart
    """
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

def plotly_box_plot(X, Y, caption, color):
    """
    Custom function for Box Plot
    """
    data = [go.Box(
            y=Y,
            x=X,
            showlegend=False,
            marker= dict(color=color)
        )]

    layout = go.Layout(
        title=caption
    )
    
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    py.iplot(fig, filename='basic-box')

def plotly_histogram_chart(X, x_title, color):
    """
    Histogram chart of Continuous variable using Plotly
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=X, marker_color=color))
    # Overlay both histograms
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(
        title_text=f"Restaurant's {x_title} Distribution", # title of plot
        xaxis_title_text=x_title, # xaxis label
        yaxis_title_text='Count', # yaxis label
        title_x=0.5,
        barmode='overlay')

    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.show()

def plotly_bubble_chart(X, Y, label):
    """
    Bubble chart using Plotly
    """
    fig = go.Figure(data=go.Scatter(
                    x=X,
                    y=Y,
                    hovertext = label,
                    mode='markers',
                    marker=dict(
                         color='rgb(255, 178, 102)',
                         size=10,
                         line=dict(
                            color='DarkSlateGrey',
                            width=1
                          )
                   )
    ))
    
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(
        title=f'Rating vs # Votes vs Cost',
        xaxis_title='Rating',
        yaxis_title='# Votes'
    )
    fig.show()
    
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    """Function to Plot Word Clouds"""

    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
```


```python
### Text Pre-Processing Related Functions ###
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

def generate_ngrams(text, n_gram=1):
    """
    Custom function for ngram generation
    """
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

def gensim_clean_text(s):
    """
    Clean the text using Gensim library
    """
    filters = [
               gsp.strip_tags, 
               gsp.strip_punctuation,
               gsp.strip_multiple_whitespaces,
               gsp.strip_numeric,
               gsp.remove_stopwords, 
               gsp.strip_short
              ]
    s = s.lower()
    s = utils.to_unicode(s)
    s = re.sub(r'[^\x00-\x7F]+',' ', s)
    s = lemmatize_words(s)
    for f in filters:
        s = f(s)
    return s
    retu
```

### Main Functions


```python
### Feature Engineering & Date Cleaning (reviews) functions ###

def _extract_ratings(text_ls):
    """
    Extract Ratings from reviews_list and taking average
    """
    try:
        rate_ls = ast.literal_eval(text_ls)
        rate_ls = [float(x[0].lower().replace("rated","").strip()) for x in rate_ls]
        final_rating = np.average(rate_ls)
        return np.round(final_rating,1)
    except Exception as e:
        return np.nan

def _extract_reviews(row):
    """
    Extract Reviews Text from reviews_list
    """
    try:
        text_ls = row["reviews_list"]
        review_ls = ast.literal_eval(text_ls)
        if len(review_ls) > 0:
            review_ls = "|".join(filter(None,[x[1] for x in review_ls]))
            no_of_reviews = len(review_ls.split("|"))
            return review_ls, no_of_reviews
        else:
            return np.nan, np.nan
    except Exception as e:
        return np.nan, np.nan

def _clean_rating(rating):
    """
    3.4/5: 3.4 (Extracting rating)
    """
    try:
        rating = float(rating.split("/")[0].strip())
        return rating
    except Exception as e:
        return np.nan

def _clean_menu_items(row):
    try:
        text_ls = row["menu_items"]
        menu_ls = ast.literal_eval(text_ls)
        if len(menu_ls) > 0:
            menu_ls = "|".join(menu_ls)
            no_of_menu = len(list(filter(None,menu_ls.split("|"))))
            return menu_ls, no_of_menu
        else:
            return np.nan, np.nan
    except Exception as e:
        return np.nan, np.nan
```


```python
def restaurant_recommend_func(df, topn=3):   
    """
    Main function for Content Based Recommender System
        a) This function first filter out the restaurant's based on User's Location, Cuisines & Budget Input Parameters
        b) After filtering out the restaurant's data, we create a tf-idf matrix for restaurant's review and then calculate
           a cosine similraity with User's Description
        c) Return's the Top 3 Restaurant's Name based on User Input's Parameters
    """
    global data_sample       
    global cosine_sim
    global sim_scores
    global tfidf_matrix
    global corpus_index
    global feature
    global rest_indices
    global idx

    ## Applying USER_LOCATION, USER_CUISINES & USER_BUDGET Filter

    data_sample = df[(df["location"]==USER_LOCATION) & (df["cuisines"].str.contains(USER_CUISINES, case=False)) &\
                           (df["cost_for_two_people"] <= USER_BUDGET)].reset_index(drop=True)
    print("Dataset Shape after applying Location, Cuisines and Budget filters: ", data_sample.shape)

    ## Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words='english')

    ## Create TF-IDF matrix from reviews (text)
    tf_idf_matrix = tfidf_vectorizer.fit_transform(data_sample["extracted_reviews"])

    ## Create TF-IDF matrix from User's Descritption Input
    user_tfidf_matrix = tfidf_vectorizer.transform([USER_DESCRIPTION])
    
    ## Using for see Cosine Similarty scores
    feature= tfidf_vectorizer.get_feature_names()
    
    ## Cosine Similarity
    ## Compute the cosine similarity matrix between user description (matrix) & created tf-idf matrix
    
    cosine_sim = linear_kernel(tf_idf_matrix, user_tfidf_matrix) 
    
    # Column names are using for index
    corpus_index=[n for n in data_sample['extracted_reviews']]
       
    # Construct a reverse map of indices    
    indices = pd.Series(data_sample.index, index=data_sample['restaurant_name']).drop_duplicates() 
    
    # rating added with cosine score in sim_score list.
    sim_scores=[]
    for i,j in enumerate(cosine_sim):
        k=data_sample['rating'].iloc[i]
        if j != 0 :
            sim_scores.append((i,j,k))
            
    # Sort the restaurant names based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: (x[1],x[2]) , reverse=True)
    
    # similar description
    sim_scores = sim_scores[0:topn]
    rest_indices = [i[0] for i in sim_scores] 
  
    data_x =data_sample[['restaurant_id', 'restaurant_name','location', 'cuisines', 'cost_for_two_people',
                         'rating']].iloc[rest_indices]
    
    data_x['Cosine Similarity']=0
    for i,j in enumerate(sim_scores):
        data_x['Cosine Similarity'].iloc[i]=sim_scores[i][1]*100
    data_x.columns = ["Restaurant ID", "Restaurant Name", "Location", "Cuisines", "Cost", "Rating", "% Likely"]
    
    return data_x
```

### Directory Setup


```python
"""
Get the working directory
"""
WORKING_DIR = os.getcwd()
```


```python
"""
Create raw, processed, cleaned directory in data directory
"""
DATA_DIR = mkdir(WORKING_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw") # All raw data will be present here
PROCESSED_DIR = mkdir(os.path.join(WORKING_DIR, DATA_DIR.split("\\")[-1]), "processed") # All processed data will be here
CLEANED_DIR = mkdir(os.path.join(WORKING_DIR, DATA_DIR.split("\\")[-1]), "cleaned") # All cleaned data will be here
```


```python
#################################################################################################
# Create output data directory: 
#################################################################################################
OUTPUT_DIR = mkdir(WORKING_DIR, "outputs")
```

### Data Preparation & Exploration

#### Load Datasets


```python
"""
Load Restaurant Training Dataset
"""
restaurant_df = load_dataframe(os.path.join(RAW_DIR, "RestoInfo.csv"))
print("There are {} Rows and {} Columns in Restaurant Training Dataset".format(restaurant_df.shape[0],restaurant_df.shape[1]))

## Remove Duplicate Rows (if any)
restaurant_df.drop_duplicates(inplace = True)
print("There are {} Rows and {} Columns in Restaurant Training Dataset After Removing Duplicates".format(restaurant_df.shape[0],
                                                                                                         restaurant_df.shape[1]))
```

    There are 2069 Rows and 15 Columns in Restaurant Training Dataset
    There are 2069 Rows and 15 Columns in Restaurant Training Dataset After Removing Duplicates
    

It is clear from above cell that there are no duplicates (rows) in Restaurant data. Next, let us look at top rows of dataframe:


```python
"""
Rename column names
"""
restaurant_df.columns = ["restaurant_id", "restaurant_name", "online_order", "book_table", "rating", "no_of_votes", "location",
                        "restaurant_type", "dish_liked", "cuisines", "cost_for_two_people", "reviews_list", "menu_items",
                        "listed_in_type", "listed_in_city"]
```


```python
"""
Check datatypes of all the features
"""
data_type_df = pd.DataFrame(restaurant_df.dtypes).reset_index().rename(columns = {'index':'Columns', 0:'Data Type'})
gradient_dataframe(data_type_df, caption="Data Types in Restaurant Dataset")
```




<style  type="text/css" >
    #T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56 tr:nth-of-type(odd) {
          background: #eee;
          text-align: left;
    }    #T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56 tr:nth-of-type(even) {
          background: white;
          text-align: left;
    }    #T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56 th {
          background: #808080;
          color: white;
          font-family: verdana;
          text-align: left;
    }    #T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56 td {
          font-family: verdana;
          text-align: left;
    }</style><table id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56" ><caption>Data Types in Restaurant Dataset</caption><thead>    <tr>        <th class="col_heading level0 col0" >Columns</th>        <th class="col_heading level0 col1" >Data Type</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row0_col0" class="data row0 col0" >restaurant_id</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row0_col1" class="data row0 col1" >int64</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row1_col0" class="data row1 col0" >restaurant_name</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row1_col1" class="data row1 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row2_col0" class="data row2 col0" >online_order</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row2_col1" class="data row2 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row3_col0" class="data row3 col0" >book_table</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row3_col1" class="data row3 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row4_col0" class="data row4 col0" >rating</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row4_col1" class="data row4 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row5_col0" class="data row5 col0" >no_of_votes</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row5_col1" class="data row5 col1" >int64</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row6_col0" class="data row6 col0" >location</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row6_col1" class="data row6 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row7_col0" class="data row7 col0" >restaurant_type</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row7_col1" class="data row7 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row8_col0" class="data row8 col0" >dish_liked</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row8_col1" class="data row8 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row9_col0" class="data row9 col0" >cuisines</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row9_col1" class="data row9 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row10_col0" class="data row10 col0" >cost_for_two_people</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row10_col1" class="data row10 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row11_col0" class="data row11 col0" >reviews_list</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row11_col1" class="data row11 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row12_col0" class="data row12 col0" >menu_items</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row12_col1" class="data row12 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row13_col0" class="data row13 col0" >listed_in_type</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row13_col1" class="data row13 col1" >object</td>
            </tr>
            <tr>
                                <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row14_col0" class="data row14 col0" >listed_in_city</td>
                        <td id="T_0c94dd98_e9c2_11ea_947c_7470fd5d5a56row14_col1" class="data row14 col1" >object</td>
            </tr>
    </tbody></table>



#### Missing Value Count


```python
complete_miss_df = missing_value_of_data(restaurant_df)
complete_miss_df = complete_miss_df.reset_index()
complete_miss_df.rename(columns = {'index':'Columns'}, inplace = True)
gradient_dataframe(complete_miss_df, caption="Missing values for Restaurant Dataset")
```




<style  type="text/css" >
    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56 tr:nth-of-type(odd) {
          background: #eee;
          text-align: left;
    }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56 tr:nth-of-type(even) {
          background: white;
          text-align: left;
    }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56 th {
          background: #808080;
          color: white;
          font-family: verdana;
          text-align: left;
    }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56 td {
          font-family: verdana;
          text-align: left;
    }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row0_col1 {
            background-color:  #08306b;
            color:  #f1f1f1;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row0_col2 {
            background-color:  #08306b;
            color:  #f1f1f1;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row1_col1 {
            background-color:  #bfd8ed;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row1_col2 {
            background-color:  #bfd8ed;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row2_col1 {
            background-color:  #f5f9fe;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row2_col2 {
            background-color:  #f5f9fe;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row3_col1 {
            background-color:  #f5f9fe;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row3_col2 {
            background-color:  #f5f9fe;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row4_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row4_col2 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row5_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row5_col2 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row6_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row6_col2 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row7_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row7_col2 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row8_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row8_col2 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row9_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row9_col2 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row10_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row10_col2 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row11_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row11_col2 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row12_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row12_col2 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row13_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row13_col2 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row14_col1 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row14_col2 {
            background-color:  #f7fbff;
            color:  #000000;
        }</style><table id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56" ><caption>Missing values for Restaurant Dataset</caption><thead>    <tr>        <th class="col_heading level0 col0" >Columns</th>        <th class="col_heading level0 col1" >Total</th>        <th class="col_heading level0 col2" >Percentage</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row0_col0" class="data row0 col0" >dish_liked</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row0_col1" class="data row0 col1" >1107</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row0_col2" class="data row0 col2" >53.500000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row1_col0" class="data row1 col0" >rating</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row1_col1" class="data row1 col1" >299</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row1_col2" class="data row1 col2" >14.450000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row2_col0" class="data row2 col0" >restaurant_type</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row2_col1" class="data row2 col1" >17</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row2_col2" class="data row2 col2" >0.820000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row3_col0" class="data row3 col0" >cost_for_two_people</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row3_col1" class="data row3 col1" >16</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row3_col2" class="data row3 col2" >0.770000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row4_col0" class="data row4 col0" >listed_in_city</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row4_col1" class="data row4 col1" >0</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row4_col2" class="data row4 col2" >0.000000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row5_col0" class="data row5 col0" >listed_in_type</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row5_col1" class="data row5 col1" >0</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row5_col2" class="data row5 col2" >0.000000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row6_col0" class="data row6 col0" >menu_items</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row6_col1" class="data row6 col1" >0</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row6_col2" class="data row6 col2" >0.000000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row7_col0" class="data row7 col0" >reviews_list</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row7_col1" class="data row7 col1" >0</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row7_col2" class="data row7 col2" >0.000000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row8_col0" class="data row8 col0" >cuisines</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row8_col1" class="data row8 col1" >0</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row8_col2" class="data row8 col2" >0.000000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row9_col0" class="data row9 col0" >location</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row9_col1" class="data row9 col1" >0</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row9_col2" class="data row9 col2" >0.000000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row10_col0" class="data row10 col0" >no_of_votes</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row10_col1" class="data row10 col1" >0</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row10_col2" class="data row10 col2" >0.000000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row11_col0" class="data row11 col0" >book_table</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row11_col1" class="data row11 col1" >0</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row11_col2" class="data row11 col2" >0.000000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row12_col0" class="data row12 col0" >online_order</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row12_col1" class="data row12 col1" >0</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row12_col2" class="data row12 col2" >0.000000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row13_col0" class="data row13 col0" >restaurant_name</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row13_col1" class="data row13 col1" >0</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row13_col2" class="data row13 col2" >0.000000</td>
            </tr>
            <tr>
                                <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row14_col0" class="data row14 col0" >restaurant_id</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row14_col1" class="data row14 col1" >0</td>
                        <td id="T_0c9e83ca_e9c2_11ea_ad98_7470fd5d5a56row14_col2" class="data row14 col2" >0.000000</td>
            </tr>
    </tbody></table>



**Key Takeaway:** Only `dish_liked`, `rating`, `restaurant_type` and `cost_for_two_people` features contains missing

#### Data Cleaning & Feature Engineering

In the given dataset we have observed the reviews_list contains rating as well as restaurant's reviews. so below code is used seperate rating & reviews.

- Calculated # Reviews/Reviews/Ratings from reviews_list
- Cleaned menu_items and calculated of # menu items


```python
"""
Cleaning reviews_list:
    a) Extracting Ratings from raw text
    b) Extracting Review (Raw Text)
    c) Extracting # of Reviews
"""

## Seperate Ratings & Text from reviews_list columns:

restaurant_df["extracted_ratings"] = restaurant_df["reviews_list"].apply(lambda x: _extract_ratings(x))
restaurant_df[["extracted_reviews", "no_of_reviews"]] = restaurant_df.apply(_extract_reviews, axis=1, result_type="expand")
```


```python
"""
cleaning rating column: 3.5/5 --> 3.5
"""
restaurant_df["rating"] = restaurant_df["rating"].apply(lambda x: _clean_rating(x))
```


```python
"""
Cleaning menu_items columns and extracting number of items
"""
restaurant_df[["menu_items", "no_of_menu_items"]] = restaurant_df.apply(_clean_menu_items, axis=1, result_type="expand")
```


```python
"""
Next, let us clean extracted_reviews column:
"""
restaurant_df["extracted_reviews"] = restaurant_df["extracted_reviews"].fillna("").progress_apply(gensim_clean_text)
```


    HBox(children=(FloatProgress(value=0.0, description='Progress', max=2069.0, style=ProgressStyle(description_wi…


    
    

#### Data Exploration

##### Most Popular Cuisines served by Restaurants


```python
cuisines_cnt_srs = restaurant_df['cuisines'].value_counts().sort_values(ascending=False).head(10)
plotly_bar_pie_chart(cuisines_cnt_srs, layout_title="Count of Top 10 Cuisines served by Restaurants",
                    pie_title="Top 10 Cuisines served by Restaurants")
```


<div>


            <div id="15700089-3e9e-4c54-9045-8c892a80ad68" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("15700089-3e9e-4c54-9045-8c892a80ad68")) {
                    Plotly.newPlot(
                        '15700089-3e9e-4c54-9045-8c892a80ad68',
                        [{"marker": {"color": [130, 111, 62, 38, 35, 33, 29, 28, 26, 25], "colorscale": [[0.0, "rgb(0,0,255)"], [0.1, "rgb(51,153,255)"], [0.2, "rgb(102,204,255)"], [0.3, "rgb(153,204,255)"], [0.4, "rgb(204,204,255)"], [0.5, "rgb(255,255,255)"], [0.6, "rgb(255,204,255)"], [0.7, "rgb(255,153,255)"], [0.8, "rgb(255,102,204)"], [0.9, "rgb(255,102,102)"], [1.0, "rgb(255,0,0)"]], "reversescale": true}, "type": "bar", "x": ["North Indian", "North Indian, Chinese", "South Indian", "Biryani", "South Indian, North Indian, Chinese", "Cafe", "Fast Food", "Bakery, Desserts", "Bakery", "Desserts"], "y": [130, 111, 62, 38, 35, 33, 29, 28, 26, 25]}],
                        {"font": {"size": 12}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Count of Top 10 Cuisines served by Restaurants"}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('15700089-3e9e-4c54-9045-8c892a80ad68');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



<div>


            <div id="f5f7d28f-5689-4462-bee6-ca9aa10d56f0" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("f5f7d28f-5689-4462-bee6-ca9aa10d56f0")) {
                    Plotly.newPlot(
                        'f5f7d28f-5689-4462-bee6-ca9aa10d56f0',
                        [{"labels": ["North Indian", "North Indian, Chinese", "South Indian", "Biryani", "South Indian, North Indian, Chinese", "Cafe", "Fast Food", "Bakery, Desserts", "Bakery", "Desserts"], "type": "pie", "values": [25.145067698259187, 21.470019342359766, 11.992263056092844, 7.35009671179884, 6.769825918762089, 6.382978723404255, 5.609284332688588, 5.415860735009671, 5.029013539651837, 4.835589941972921]}],
                        {"font": {"size": 12}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Top 10 Cuisines served by Restaurants"}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('f5f7d28f-5689-4462-bee6-ca9aa10d56f0');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** `North Indian` Cuisines is the most served by Restaurants (25.1%, 130)

##### Most Popular Restaurant's Locations


```python
city_cnt_srs = restaurant_df['location'].value_counts().sort_values(ascending=False).head(10)
plotly_bar_pie_chart(city_cnt_srs, layout_title="Count of Top 10 Restaurant's Locations",
                    pie_title="Top 10 Restaurant's Locations (City)")
```


<div>


            <div id="abe82fd5-70ac-4843-af00-51fc59f03703" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("abe82fd5-70ac-4843-af00-51fc59f03703")) {
                    Plotly.newPlot(
                        'abe82fd5-70ac-4843-af00-51fc59f03703',
                        [{"marker": {"color": [227, 99, 99, 98, 91, 79, 78, 72, 65, 51], "colorscale": [[0.0, "rgb(0,0,255)"], [0.1, "rgb(51,153,255)"], [0.2, "rgb(102,204,255)"], [0.3, "rgb(153,204,255)"], [0.4, "rgb(204,204,255)"], [0.5, "rgb(255,255,255)"], [0.6, "rgb(255,204,255)"], [0.7, "rgb(255,153,255)"], [0.8, "rgb(255,102,204)"], [0.9, "rgb(255,102,102)"], [1.0, "rgb(255,0,0)"]], "reversescale": true}, "type": "bar", "x": ["BTM", "JP Nagar", "HSR", "Koramangala 5th Block", "Marathahalli", "Whitefield", "Indiranagar", "Bannerghatta Road", "Jayanagar", "Bellandur"], "y": [227, 99, 99, 98, 91, 79, 78, 72, 65, 51]}],
                        {"font": {"size": 12}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Count of Top 10 Restaurant's Locations"}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('abe82fd5-70ac-4843-af00-51fc59f03703');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



<div>


            <div id="2d94f32c-e3bd-4b4f-af1f-1045e807312b" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("2d94f32c-e3bd-4b4f-af1f-1045e807312b")) {
                    Plotly.newPlot(
                        '2d94f32c-e3bd-4b4f-af1f-1045e807312b',
                        [{"labels": ["BTM", "JP Nagar", "HSR", "Koramangala 5th Block", "Marathahalli", "Whitefield", "Indiranagar", "Bannerghatta Road", "Jayanagar", "Bellandur"], "type": "pie", "values": [23.670490093847757, 10.323253388946819, 10.323253388946819, 10.218978102189782, 9.48905109489051, 8.237747653806048, 8.13347236704901, 7.5078206465067785, 6.777893639207508, 5.318039624608968]}],
                        {"font": {"size": 12}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Top 10 Restaurant's Locations (City)"}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('2d94f32c-e3bd-4b4f-af1f-1045e807312b');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** `BTM` is the most popular Restaurant's location (23.7%, 227)

##### Most Popular Restaurant's Type


```python
type_cnt_srs = restaurant_df['restaurant_type'].value_counts().sort_values(ascending=False).head(10)
plotly_bar_pie_chart(type_cnt_srs, layout_title="Count of Top 10 Restaurant's Type",
                    pie_title="Top 10 Restaurant's Type")
```


<div>


            <div id="0ca6b717-0046-4410-953d-348ad0c8d5c9" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("0ca6b717-0046-4410-953d-348ad0c8d5c9")) {
                    Plotly.newPlot(
                        '0ca6b717-0046-4410-953d-348ad0c8d5c9',
                        [{"marker": {"color": [796, 413, 140, 90, 88, 70, 41, 35, 34, 32], "colorscale": [[0.0, "rgb(0,0,255)"], [0.1, "rgb(51,153,255)"], [0.2, "rgb(102,204,255)"], [0.3, "rgb(153,204,255)"], [0.4, "rgb(204,204,255)"], [0.5, "rgb(255,255,255)"], [0.6, "rgb(255,204,255)"], [0.7, "rgb(255,153,255)"], [0.8, "rgb(255,102,204)"], [0.9, "rgb(255,102,102)"], [1.0, "rgb(255,0,0)"]], "reversescale": true}, "type": "bar", "x": ["Quick Bites", "Casual Dining", "Cafe", "Dessert Parlor", "Delivery", "Takeaway, Delivery", "Bakery", "Beverage Shop", "Casual Dining, Bar", "Food Court"], "y": [796, 413, 140, 90, 88, 70, 41, 35, 34, 32]}],
                        {"font": {"size": 12}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Count of Top 10 Restaurant's Type"}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('0ca6b717-0046-4410-953d-348ad0c8d5c9');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



<div>


            <div id="64ece375-fde0-4d4f-ab15-b1110a03f12f" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("64ece375-fde0-4d4f-ab15-b1110a03f12f")) {
                    Plotly.newPlot(
                        '64ece375-fde0-4d4f-ab15-b1110a03f12f',
                        [{"labels": ["Quick Bites", "Casual Dining", "Cafe", "Dessert Parlor", "Delivery", "Takeaway, Delivery", "Bakery", "Beverage Shop", "Casual Dining, Bar", "Food Court"], "type": "pie", "values": [45.77343300747556, 23.749281196089704, 8.050603795284646, 5.175388154111558, 5.060379528464635, 4.025301897642323, 2.3576768257619323, 2.0126509488211615, 1.9551466359976999, 1.8401380103507763]}],
                        {"font": {"size": 12}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Top 10 Restaurant's Type"}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('64ece375-fde0-4d4f-ab15-b1110a03f12f');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** `Quick Bites` is the most popular Restaurant's Type (23.7%, 227)

##### Dishes Liked vs Online Order


```python
like_online_df = restaurant_df[["dish_liked", "online_order"]].fillna("")
like_online_df["dish_liked"] = like_online_df["dish_liked"].str.split(", ")
like_online_df = like_online_df.explode("dish_liked")
like_online_df = like_online_df[~like_online_df["dish_liked"].isin(["", " "])]
like_online_df[like_online_df["online_order"]=="Yes"]["dish_liked"].value_counts()

yes_df = like_online_df[like_online_df["online_order"]=="Yes"]
no_df = like_online_df[like_online_df["online_order"]=="No"]

yes_sorted = pd.DataFrame(yes_df['dish_liked'].value_counts().sort_values(ascending=False)).reset_index()
yes_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(yes_sorted.head(50), 'blue')

no_sorted = pd.DataFrame(no_df['dish_liked'].value_counts().sort_values(ascending=False)).reset_index()
no_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(no_sorted.head(50), 'blue')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Online Order (Yes)", 
                                          "Online Order (No)"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Most Dishes Liked vs Online Order")
py.iplot(fig, filename='word-plots')
```

    C:\Users\jj18826\AppData\Roaming\Python\Python36\site-packages\plotly\tools.py:465: DeprecationWarning:
    
    plotly.tools.make_subplots is deprecated, please use plotly.subplots.make_subplots instead
    
    


<div>


            <div id="1ec6058d-40e3-479a-82cb-06a437f91456" class="plotly-graph-div" style="height:1200px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("1ec6058d-40e3-479a-82cb-06a437f91456")) {
                    Plotly.newPlot(
                        '1ec6058d-40e3-479a-82cb-06a437f91456',
                        [{"marker": {"color": "blue"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [14, 15, 15, 15, 16, 17, 17, 17, 18, 19, 19, 19, 20, 20, 22, 22, 22, 22, 23, 23, 24, 24, 24, 25, 26, 26, 28, 28, 28, 29, 29, 30, 30, 30, 33, 36, 36, 40, 42, 43, 46, 47, 50, 58, 59, 67, 70, 82, 83, 91], "xaxis": "x", "y": ["Rasmalai", "Chai", "Naan", "Chicken Kebab", "French Fries", "Samosa", "Hyderabadi Biryani", "Raita", "Sandwich", "Dal Makhani", "Shawarma", "Appam", "Gulab Jamun", "Hot Chocolate", "Chicken Grill", "Tiramisu", "Chilli Chicken", "Paneer Tikka", "Waffles", "Vegetable Biryani", "Lassi", "Brownie", "Tea", "Roti", "Salads", "Beer", "Tandoori Chicken", "Thali", "Salad", "Chicken Curry", "Nachos", "Sea Food", "Fries", "Chaat", "Momos", "Mutton Biryani", "Rolls", "Sandwiches", "Butter Chicken", "Fish", "Noodles", "Chicken Biryani", "Cocktails", "Coffee", "Mocktails", "Paratha", "Biryani", "Pizza", "Burgers", "Pasta"], "yaxis": "y"}, {"marker": {"color": "blue"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 16, 16, 23, 26, 27, 28, 29, 32, 37, 40, 51, 70], "xaxis": "x2", "y": ["Roti", "Pav Bhaji", "Sunday Brunch", "Dhokla", "Chocolate Cake", "Paneer Tikka", "Cheesecake", "Sandwich", "Samosa", "Jalebi", "Vegetable Biryani", "Chicken Wings", "Raita", "Momos", "Long Island Iced Tea", "Buttermilk", "Sea Food", "Dal Makhani", "Mutton Biryani", "French Fries", "Breakfast Buffet", "Kulfi", "Tandoori Chicken", "Tiramisu", "Waffles", "Butter Chicken", "Thali", "Chicken Biryani", "Tea", "Salad", "Gulab Jamun", "Panipuri", "Chaat", "Paratha", "Hot Chocolate", "Lunch Buffet", "Mocktails", "Noodles", "Brownie", "Fish", "Nachos", "Burgers", "Sandwiches", "Biryani", "Coffee", "Salads", "Beer", "Pizza", "Pasta", "Cocktails"], "yaxis": "y2"}],
                        {"annotations": [{"font": {"size": 16}, "showarrow": false, "text": "Online Order (Yes)", "x": 0.225, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}, {"font": {"size": 16}, "showarrow": false, "text": "Online Order (No)", "x": 0.775, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}], "height": 1200, "paper_bgcolor": "rgb(233,233,233)", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Most Dishes Liked vs Online Order"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 0.45]}, "xaxis2": {"anchor": "y2", "domain": [0.55, 1.0]}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0]}, "yaxis2": {"anchor": "x2", "domain": [0.0, 1.0]}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('1ec6058d-40e3-479a-82cb-06a437f91456');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
from plotly import subplots
```

**Key Takeaway:** `Pasta` is preferable in Online order and `Cocatails` is preferable in Offline Order

##### Cuisines Liked vs Online Order


```python
cuisine_online_df = restaurant_df[["cuisines", "online_order"]].fillna("")
cuisine_online_df["cuisines"] = cuisine_online_df["cuisines"].str.split(", ")
cuisine_online_df = cuisine_online_df.explode("cuisines")
cuisine_online_df = cuisine_online_df[~cuisine_online_df["cuisines"].isin(["", " "])]

yes_df = cuisine_online_df[cuisine_online_df["online_order"]=="Yes"]
no_df = cuisine_online_df[cuisine_online_df["online_order"]=="No"]

yes_sorted = pd.DataFrame(yes_df['cuisines'].value_counts().sort_values(ascending=False)).reset_index()
yes_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(yes_sorted.head(50), 'orange')

no_sorted = pd.DataFrame(no_df['cuisines'].value_counts().sort_values(ascending=False)).reset_index()
no_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(no_sorted.head(50), 'orange')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Online Order (Yes)", 
                                          "Online Order (No)"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Most Cuisines Liked vs Online Order")
py.iplot(fig, filename='word-plots')
```


<div>


            <div id="0bf70145-7dbb-400d-bf98-9c834b662b68" class="plotly-graph-div" style="height:1200px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("0bf70145-7dbb-400d-bf98-9c834b662b68")) {
                    Plotly.newPlot(
                        '0bf70145-7dbb-400d-bf98-9c834b662b68',
                        [{"marker": {"color": "orange"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [4, 5, 5, 6, 6, 6, 6, 7, 8, 9, 10, 12, 13, 13, 14, 15, 17, 17, 22, 22, 22, 23, 24, 27, 28, 31, 31, 36, 39, 43, 44, 45, 47, 48, 49, 53, 53, 59, 65, 73, 86, 115, 126, 130, 134, 158, 172, 192, 399, 525], "xaxis": "x", "y": ["Lebanese", "Malaysian", "Rajasthani", "Vietnamese", "Middle Eastern", "Turkish", "Tibetan", "Oriya", "Japanese", "Tea", "Mediterranean", "Steak", "Hyderabadi", "BBQ", "Mexican", "European", "Mangalorean", "Bengali", "Finger Food", "Healthy Food", "Mithai", "Juices", "Sandwich", "Thai", "Kerala", "Arabian", "Kebab", "Asian", "Salad", "Seafood", "Momos", "Rolls", "American", "Mughlai", "Andhra", "Ice Cream", "Bakery", "Pizza", "Burger", "Street Food", "Italian", "Desserts", "Cafe", "Beverages", "Continental", "South Indian", "Biryani", "Fast Food", "Chinese", "North Indian"], "yaxis": "y"}, {"marker": {"color": "orange"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 10, 11, 11, 12, 13, 13, 14, 15, 15, 15, 15, 15, 17, 19, 20, 21, 21, 22, 24, 24, 26, 37, 39, 49, 56, 60, 72, 80, 90, 93, 135, 159, 234, 360], "xaxis": "x2", "y": ["Gujarati", "Afghan", "Coffee", "Oriya", "Rajasthani", "Spanish", "Tibetan", "Mexican", "Chettinad", "Hyderabadi", "Mangalorean", "Momos", "Thai", "Japanese", "BBQ", "Steak", "Tea", "Healthy Food", "Bengali", "Mediterranean", "Arabian", "Sandwich", "European", "Burger", "Juices", "Rolls", "Salad", "American", "Mithai", "Asian", "Andhra", "Kebab", "Kerala", "Seafood", "Pizza", "Ice Cream", "Mughlai", "Finger Food", "Italian", "Street Food", "Beverages", "Bakery", "Cafe", "Continental", "Desserts", "Biryani", "Fast Food", "South Indian", "Chinese", "North Indian"], "yaxis": "y2"}],
                        {"annotations": [{"font": {"size": 16}, "showarrow": false, "text": "Online Order (Yes)", "x": 0.225, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}, {"font": {"size": 16}, "showarrow": false, "text": "Online Order (No)", "x": 0.775, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}], "height": 1200, "paper_bgcolor": "rgb(233,233,233)", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Most Cuisines Liked vs Online Order"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 0.45]}, "xaxis2": {"anchor": "y2", "domain": [0.55, 1.0]}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0]}, "yaxis2": {"anchor": "x2", "domain": [0.0, 1.0]}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('0bf70145-7dbb-400d-bf98-9c834b662b68');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** `South Indian` is the more preferable in Offline Order then Online Order

##### Location's with most votes and best ratings for their Restaurants


```python
city_business_reviews = restaurant_df[['listed_in_city', 'no_of_votes', 'rating']].groupby(['listed_in_city'], 
                                                                                          as_index = False).\
agg({'no_of_votes': 'sum', 'rating': 'mean'})

votes_df = city_business_reviews[["listed_in_city", "no_of_votes"]]
rating_df = city_business_reviews[["listed_in_city", "rating"]]

votes_sorted = votes_df.sort_values(['no_of_votes'], ascending=False)
votes_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(votes_sorted.head(50), 'red')

rating_sorted = rating_df.sort_values(['rating'], ascending=False)
rating_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(rating_sorted.head(50), 'green')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["By Votes", 
                                          "By Rating"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Top Location's by Votes & Reviews")
py.iplot(fig, filename='word-plots')
```


<div>


            <div id="68fd5575-f215-4ebf-af66-8fa5b88dfc64" class="plotly-graph-div" style="height:1200px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("68fd5575-f215-4ebf-af66-8fa5b88dfc64")) {
                    Plotly.newPlot(
                        '68fd5575-f215-4ebf-af66-8fa5b88dfc64',
                        [{"marker": {"color": "red"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [2261, 3372, 4118, 4211, 5139, 7181, 8019, 8784, 9382, 9566, 10095, 11638, 11648, 12623, 13206, 15070, 17619, 19914, 21401, 23027, 24296, 26947, 27643, 31280, 32760, 33159, 40367, 42783, 42853, 50782], "xaxis": "x", "y": ["New BEL Road", "Electronic City", "Kalyan Nagar", "Rajajinagar", "Banashankari", "Kammanahalli", "Brookefield", "Frazer Town", "Malleshwaram", "Bannerghatta Road", "Bellandur", "Whitefield", "HSR", "Sarjapur Road", "Marathahalli", "JP Nagar", "Old Airport Road", "Residency Road", "Basavanagudi", "MG Road", "Jayanagar", "Lavelle Road", "Church Street", "Brigade Road", "Indiranagar", "BTM", "Koramangala 5th Block", "Koramangala 7th Block", "Koramangala 6th Block", "Koramangala 4th Block"], "yaxis": "y"}, {"marker": {"color": "green"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [3.3207547169811322, 3.5127659574468084, 3.5209302325581397, 3.528333333333332, 3.552173913043478, 3.5555555555555554, 3.5628571428571427, 3.566, 3.597435897435897, 3.6508771929824566, 3.6641025641025635, 3.6666666666666665, 3.666666666666668, 3.66909090909091, 3.6824561403508773, 3.683561643835617, 3.702272727272728, 3.723255813953489, 3.728947368421054, 3.7436619718309867, 3.7439024390243887, 3.7476923076923065, 3.7499999999999996, 3.7725490196078417, 3.777659574468087, 3.780357142857144, 3.787755102040817, 3.7967213114754106, 3.8078651685393257, 3.9072727272727263], "xaxis": "x2", "y": ["Brookefield", "Whitefield", "Bellandur", "Marathahalli", "New BEL Road", "Electronic City", "Sarjapur Road", "Bannerghatta Road", "Kalyan Nagar", "Malleshwaram", "Kammanahalli", "Rajajinagar", "BTM", "HSR", "JP Nagar", "Indiranagar", "Old Airport Road", "Koramangala 7th Block", "Frazer Town", "Jayanagar", "Basavanagudi", "MG Road", "Banashankari", "Residency Road", "Koramangala 5th Block", "Lavelle Road", "Koramangala 6th Block", "Brigade Road", "Koramangala 4th Block", "Church Street"], "yaxis": "y2"}],
                        {"annotations": [{"font": {"size": 16}, "showarrow": false, "text": "By Votes", "x": 0.225, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}, {"font": {"size": 16}, "showarrow": false, "text": "By Rating", "x": 0.775, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}], "height": 1200, "paper_bgcolor": "rgb(233,233,233)", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Top Location's by Votes & Reviews"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 0.45]}, "xaxis2": {"anchor": "y2", "domain": [0.55, 1.0]}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0]}, "yaxis2": {"anchor": "x2", "domain": [0.0, 1.0]}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('68fd5575-f215-4ebf-af66-8fa5b88dfc64');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** `Koramangala 4th Block` is most the voted location and `Church Street` is most the rated location

##### Restaurant's Type with most votes and best ratings


```python
restaurant_business_reviews = restaurant_df[['listed_in_type', 'no_of_votes', 'rating']].groupby(['listed_in_type'], 
                                                                                          as_index = False).\
agg({'no_of_votes': 'sum', 'rating': 'mean'})

votes_df = restaurant_business_reviews[["listed_in_type", "no_of_votes"]]
rating_df = restaurant_business_reviews[["listed_in_type", "rating"]]

votes_sorted = votes_df.sort_values(['no_of_votes'], ascending=False)
votes_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(votes_sorted.head(50), 'red')

rating_sorted = rating_df.sort_values(['rating'], ascending=False)
rating_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(rating_sorted.head(50), 'green')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["By Votes", 
                                          "By Rating"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Top Restaurant's Type by Votes & Reviews")
py.iplot(fig, filename='word-plots')
```


<div>


            <div id="7f07c65e-ef92-4ec1-a27b-d9d54d79758c" class="plotly-graph-div" style="height:1200px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("7f07c65e-ef92-4ec1-a27b-d9d54d79758c")) {
                    Plotly.newPlot(
                        '7f07c65e-ef92-4ec1-a27b-d9d54d79758c',
                        [{"marker": {"color": "red"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [12130, 14087, 18756, 22236, 55951, 218328, 229656], "xaxis": "x", "y": ["Cafes", "Desserts", "Buffet", "Pubs and bars", "Drinks & nightlife", "Dine-out", "Delivery"], "yaxis": "y"}, {"marker": {"color": "green"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [3.645283018867926, 3.68998272884283, 3.71764705882353, 3.7377358490566035, 3.9148148148148145, 4.015384615384615, 4.037777777777778], "xaxis": "x2", "y": ["Delivery", "Dine-out", "Desserts", "Cafes", "Buffet", "Pubs and bars", "Drinks & nightlife"], "yaxis": "y2"}],
                        {"annotations": [{"font": {"size": 16}, "showarrow": false, "text": "By Votes", "x": 0.225, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}, {"font": {"size": 16}, "showarrow": false, "text": "By Rating", "x": 0.775, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}], "height": 1200, "paper_bgcolor": "rgb(233,233,233)", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Top Restaurant's Type by Votes & Reviews"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 0.45]}, "xaxis2": {"anchor": "y2", "domain": [0.55, 1.0]}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0]}, "yaxis2": {"anchor": "x2", "domain": [0.0, 1.0]}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('7f07c65e-ef92-4ec1-a27b-d9d54d79758c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** `Delivery` restaurant's are most voted and `Drinks & nightlife` restaurant's are most rated

##### Book Table vs Rating


```python
plotly_box_plot(restaurant_df["book_table"], restaurant_df["rating"], 
                caption="Book Table vs Rating", color="Blue")
```


<div>


            <div id="141a8cb9-381b-425f-ae10-7c2e1a907801" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("141a8cb9-381b-425f-ae10-7c2e1a907801")) {
                    Plotly.newPlot(
                        '141a8cb9-381b-425f-ae10-7c2e1a907801',
                        [{"marker": {"color": "Blue"}, "showlegend": false, "type": "box", "x": ["No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No"], "y": [null, 3.1, 4.0, 4.2, 3.9, 4.1, 3.3, 4.4, 3.9, 3.8, 3.3, null, 3.1, 3.5, 3.8, 2.9, 4.1, null, 3.9, 4.1, 3.8, null, 4.4, null, 4.1, 3.7, null, 3.5, null, 3.4, 3.7, 4.5, null, 4.3, 3.6, 2.8, 3.6, 3.8, 4.0, 3.8, 3.3, 3.4, 3.5, 4.4, null, 4.3, 3.2, 4.0, 2.9, null, 3.1, 4.0, 3.0, 3.4, 4.4, null, 3.9, 4.4, null, 3.2, null, 4.1, 4.6, 3.6, 3.6, 3.9, 3.4, 4.1, null, 3.5, 3.7, 3.7, 3.9, null, 3.9, 4.0, 3.8, 2.9, 3.9, 3.8, 2.8, 3.6, 2.8, 4.0, 3.7, 3.8, null, null, 3.1, 2.9, 3.9, 3.2, 4.0, 3.7, 3.0, null, 4.8, null, null, 3.6, 3.6, null, 3.3, 3.7, null, 4.1, 3.8, null, 4.0, 3.8, 4.0, 3.5, null, 4.0, 3.6, 2.9, 4.5, 4.4, 2.8, 3.6, 3.8, 4.4, 4.1, 4.3, 3.8, 3.7, 3.7, null, null, null, 2.9, 3.4, 3.5, 3.0, 3.8, 3.9, 3.5, 3.1, 3.0, null, 3.8, null, 3.1, 4.0, 3.8, 3.4, 3.7, 3.4, 3.0, 4.4, 4.1, 4.4, 4.0, null, null, 3.3, 3.8, 2.9, 4.2, null, 3.0, null, null, null, 4.1, 3.9, 3.6, 3.8, 2.1, 3.6, null, 4.2, 3.8, 3.9, null, null, 3.5, null, 3.2, 3.5, 3.6, null, 4.4, 3.6, 4.2, null, 4.1, 3.6, 4.0, null, 3.8, 4.1, 4.4, 3.7, 3.1, 4.0, 3.7, 3.7, 3.4, 4.3, 3.3, 4.4, null, 4.1, 3.7, 3.4, 3.6, null, 4.1, 3.6, 3.8, 4.2, null, null, 4.1, 3.1, 4.4, 3.1, null, null, 3.9, null, 3.9, null, null, 3.0, 3.8, 4.0, 4.0, 4.9, 3.0, 4.0, 3.9, 3.8, 4.0, 4.3, 3.9, 4.2, 4.0, null, 4.0, 4.1, null, 3.7, 4.1, 4.2, 3.5, 3.3, 3.9, 3.9, 3.9, null, 3.1, null, 4.3, 4.2, null, 3.7, 4.1, null, 3.3, 3.0, 3.2, 3.4, 4.3, 4.3, 3.9, 3.6, null, 4.0, 3.5, null, 3.9, 3.4, 3.7, 4.1, 3.4, 3.6, 3.1, null, 3.4, 3.9, 3.7, 4.1, 3.7, 4.2, 2.8, 4.3, 3.8, 4.0, null, null, 3.7, 3.5, 3.7, 4.3, 3.6, 4.0, 4.3, null, null, 3.5, null, 3.8, 3.3, 4.0, 3.8, 3.5, 3.9, 4.1, 3.8, 3.4, 3.9, 3.8, 3.8, 4.4, 3.2, 3.7, 3.8, 3.6, 4.1, 3.3, 3.5, 3.3, 3.7, 4.4, 3.6, 3.7, 2.7, null, 3.5, 3.7, 4.0, 3.7, 4.2, 3.7, 4.2, null, 3.5, 3.6, 3.5, 4.1, 4.0, 4.0, 4.3, 3.8, 3.8, null, 3.8, 3.7, null, 3.7, 3.5, 3.6, 4.0, 2.8, 4.1, 3.7, 3.9, 3.5, 4.1, null, 4.2, 4.0, null, 3.7, 4.3, 4.0, 4.0, 3.7, 3.5, 3.6, 4.0, 4.4, 3.9, null, 3.2, 4.2, 4.1, 4.3, null, 3.3, 4.1, 4.1, null, null, 3.5, 4.2, 3.6, 4.1, 3.9, 4.0, null, null, 3.9, 2.8, 4.1, 3.5, null, null, 3.8, 3.5, 4.5, 3.6, 3.2, null, 3.9, null, 3.5, 3.3, 3.9, 3.0, 4.3, 3.9, 3.8, 3.5, null, 3.4, null, 3.5, null, 3.7, 4.1, 3.8, 3.2, 2.8, 3.5, 3.3, null, null, 3.4, 4.0, null, 3.5, 3.9, 3.7, 3.8, 4.0, 3.2, 4.2, 4.3, 3.7, 4.0, 2.8, 3.3, 3.4, 3.3, 3.7, 3.8, 3.7, 3.3, 3.7, null, 4.1, 4.2, 4.0, 4.0, 3.6, null, 3.7, 4.2, 3.2, 3.8, 3.5, 3.3, 3.8, null, null, null, 3.4, 3.8, 3.8, 4.1, 3.2, 3.8, 4.0, 3.7, 3.0, 3.2, 4.4, null, 3.3, 3.2, 4.0, 3.7, null, 4.1, 2.6, null, 4.1, 4.1, null, 3.6, 3.8, 2.8, null, 2.9, 4.0, 3.0, 3.7, 4.3, null, 3.9, 3.5, 4.2, 4.0, null, 3.6, 3.5, 3.7, 3.8, 3.4, 4.2, 3.6, 4.5, 4.1, 3.7, 3.9, 3.7, 4.5, 3.6, 4.0, null, 3.7, null, 4.0, 3.9, 4.1, 4.2, null, 3.4, 3.3, null, null, 2.7, 3.2, null, 3.4, 3.7, null, 4.0, 3.4, null, 3.7, 2.7, null, 4.3, null, 4.3, 3.7, 3.5, 4.3, null, 3.6, 2.2, 4.2, 3.8, null, 4.3, 4.1, 3.1, 3.1, 3.9, 4.4, 3.9, 3.5, 3.4, 2.8, 3.8, null, 3.8, null, 3.0, 4.2, null, 2.9, 4.2, 4.4, null, 3.3, 3.3, 4.0, 3.9, null, 4.2, 3.2, null, 3.5, 3.1, null, null, 4.1, null, 3.8, 4.6, null, null, 4.2, 3.2, 3.5, 3.7, 3.8, 3.8, 3.4, 4.0, 3.3, 3.2, 3.7, 3.5, null, 3.8, 3.5, 3.5, 4.3, null, 3.9, 3.8, 4.1, 4.2, 3.6, 3.9, 3.9, 3.1, 3.5, 3.7, null, 4.2, 4.5, 3.7, null, 4.0, null, 3.2, 4.1, 4.1, 4.3, 3.4, 4.0, 4.1, 3.5, 3.8, 3.6, 3.9, 2.7, 4.3, 4.0, 3.8, 4.1, 3.9, 3.5, 3.6, 3.9, 3.5, 4.1, 3.6, 3.2, 3.2, 3.7, null, null, 3.4, 3.6, null, 4.3, 2.4, 3.5, 3.3, 3.7, 3.2, null, 3.0, 4.0, 4.0, 3.9, 3.5, 3.9, 3.4, 3.7, null, 4.0, 3.1, null, 3.2, 3.6, 3.6, 3.7, 4.1, 3.7, 3.6, 4.3, 2.9, 4.2, 3.8, null, 4.4, 3.5, 4.0, 3.3, 3.7, 3.8, 3.4, 4.0, 3.5, null, 2.9, 3.3, 3.7, 3.9, 3.6, 3.9, 3.3, 3.4, 3.7, 4.4, 2.8, 3.5, 3.9, 2.9, null, 4.0, null, 3.8, 2.8, 4.3, 3.6, 2.8, 3.5, 3.2, 3.8, 3.3, null, 2.9, 4.2, 2.7, null, 3.4, null, 3.7, 4.1, null, 4.4, 2.7, null, 3.6, 3.1, null, 4.3, 2.8, 3.8, 3.5, 3.2, 3.6, 3.9, 3.9, 3.6, 3.9, 4.1, null, null, 4.5, 3.3, 3.8, null, null, 3.3, 3.6, 3.5, 3.3, 4.1, null, 4.2, 3.9, 3.7, 3.9, 3.8, 3.6, 3.9, 4.4, 3.7, 3.5, 4.0, null, 3.8, 4.0, 2.9, 3.9, 3.9, 4.1, 3.7, 3.5, 3.1, null, 4.4, 3.7, 2.9, 2.6, 4.0, 2.5, 3.6, 3.3, 3.8, 3.7, 3.5, 2.8, 3.9, 3.8, 3.7, 3.0, null, 3.2, null, 4.4, 3.4, 3.7, 3.5, null, 4.3, 4.3, null, 3.4, 3.3, 2.8, 3.4, 4.2, 3.9, 3.8, 4.4, null, 4.1, 3.6, 3.4, 3.0, 3.7, 2.1, 3.7, 3.3, 3.9, null, 3.0, null, 3.0, 3.1, 2.8, 3.5, 3.4, 3.3, 4.1, null, 3.9, null, 4.1, 3.7, 3.8, 3.8, 3.0, null, 3.9, 3.3, 4.4, 3.5, 3.7, 4.3, 3.7, 4.2, 3.6, null, 4.1, null, 4.0, 3.4, 2.7, 2.9, null, 4.4, 3.6, 4.0, null, 3.1, 3.4, null, 3.3, 3.8, 3.4, 3.4, null, 4.2, 4.1, null, null, null, 3.2, 3.4, 3.9, 4.1, 2.6, 3.9, 3.0, 3.6, 4.0, 3.7, 4.3, null, null, 4.1, 3.4, 4.3, 3.6, 4.5, 3.9, 3.8, 4.1, 3.6, 3.3, 3.7, 3.8, 3.7, 3.9, 3.6, 4.2, null, 4.1, 3.9, null, 4.2, 3.8, 4.4, 3.7, null, null, 3.9, 2.9, 3.3, 3.1, 3.8, 4.4, 3.2, 3.6, 4.3, 4.1, 3.8, 2.9, 4.1, 3.1, 3.4, null, 3.9, 3.7, null, 4.0, 2.8, 3.3, 3.8, null, 4.0, 2.2, 4.0, null, 4.2, 3.4, 3.3, 3.6, 3.3, 3.6, 3.8, 2.8, 3.6, 3.8, null, null, 3.5, 4.0, null, null, 4.3, 3.3, 3.4, 3.8, 3.7, 3.6, 4.5, null, 3.0, 3.1, 3.6, 3.7, 3.7, null, 4.0, null, 4.2, null, null, null, 3.7, 3.2, 4.3, 4.5, null, 3.2, 3.5, 3.9, 3.8, 3.5, 4.1, 3.1, 4.2, null, 3.5, 4.4, 3.8, 4.0, null, 3.1, 3.8, null, 3.9, null, 3.9, 3.8, null, null, 3.6, 3.5, 4.2, 3.9, null, 1.8, 4.0, 4.0, null, 4.1, null, 3.5, 4.1, 2.6, null, 3.7, 3.5, null, 3.8, 4.5, null, 3.3, 3.2, 2.9, null, null, 4.2, 3.7, 3.9, 3.8, 4.0, 3.9, 3.1, 4.3, 4.3, 4.7, 3.8, 2.9, 4.1, null, 4.1, 4.2, null, 3.8, 4.4, 3.8, 3.7, 3.8, 4.0, null, 2.8, null, 4.4, 3.7, null, 3.4, 4.0, 4.1, null, null, 3.7, 2.9, null, 3.1, 4.4, 3.3, 3.3, 4.0, 3.8, 3.6, null, 4.4, 4.6, 4.3, 3.3, 3.8, 4.0, null, 4.1, 3.9, 3.8, 3.7, 3.3, 4.2, 4.2, 2.6, null, 3.5, 3.1, 2.8, null, 4.1, 4.0, 3.1, 3.4, 3.5, 3.4, 4.3, 3.7, null, null, null, 3.9, null, 4.3, 3.9, 4.1, 3.8, 4.3, 2.6, 3.6, 3.4, 4.2, null, 4.0, null, 3.7, 3.6, 3.6, 3.6, 4.4, 3.5, 4.4, 3.5, 4.5, null, null, 3.6, 3.9, 3.1, null, 4.4, 3.4, null, 3.9, 3.8, 3.7, null, 4.1, 3.4, 3.7, 4.0, 3.5, 4.2, 4.4, 3.5, null, 3.9, 3.2, 4.0, 3.5, 3.4, 4.4, 3.7, 4.0, null, 4.1, null, 2.8, 3.9, 3.8, 4.0, 2.8, 3.7, 3.5, null, 2.3, null, null, 4.1, 3.7, 3.7, 2.8, 3.3, 3.0, null, null, 3.6, 4.2, 3.8, 3.1, 3.0, 4.2, 3.1, 3.3, 3.7, null, null, 3.8, 4.1, 3.9, null, 4.1, null, 3.7, 3.5, 3.7, 3.7, 3.8, 3.5, 2.8, 3.9, 3.2, 3.2, 4.0, 3.1, 3.9, null, 4.2, 4.4, 4.4, 2.8, 4.2, 2.8, 3.1, 3.7, 2.8, 4.0, 3.8, 3.6, 2.9, 3.8, 4.2, 4.5, 3.6, null, null, null, 3.7, null, 3.2, 3.4, null, 3.2, 4.3, 3.8, null, 3.7, 3.4, 4.0, 4.0, 3.4, 2.5, 3.3, 3.8, 4.1, 3.4, 4.5, 3.8, 4.6, 3.1, 4.4, 3.2, 2.7, 3.7, 3.6, 3.4, null, 2.9, null, 4.0, 3.2, null, 4.0, 3.8, 4.2, 3.5, 2.8, null, 3.8, 4.6, null, 3.4, 3.0, null, null, null, null, 4.1, 4.5, 4.2, 3.5, 3.4, null, null, null, 3.0, 3.3, 3.2, 3.9, 3.6, null, 4.2, 4.0, 2.9, null, 3.4, 3.4, null, 3.8, 3.8, 3.7, 4.0, 3.6, 4.2, 3.7, 4.5, 4.1, 3.2, null, 4.0, 3.0, 4.0, 3.2, 2.9, 4.3, 4.0, 4.7, null, 4.0, 3.2, 4.0, 3.9, 4.1, 3.3, 3.7, null, null, 3.8, null, 3.1, 4.0, 3.6, null, 3.6, 3.8, 3.4, 3.3, 3.2, 4.5, null, null, 4.1, 3.6, 4.2, 3.0, null, null, 4.3, 4.1, 2.8, 3.6, 3.1, 3.8, 4.0, 4.1, null, 4.2, 4.1, 3.7, 2.6, 3.2, 3.3, 2.8, 3.9, null, 3.9, 3.1, 3.8, null, 3.1, 4.3, null, 4.3, 3.6, 3.9, null, 3.2, 3.6, null, 3.6, null, 4.1, 4.0, 4.3, 3.3, 3.3, null, null, 3.2, 3.5, 3.6, 3.5, 2.9, 3.9, 3.9, 3.4, 4.2, 3.8, 3.4, 3.6, 4.0, 4.1, 3.6, 4.2, 3.7, 2.7, 3.7, 3.3, 3.1, null, 2.8, 3.9, 4.3, null, 3.3, 3.7, null, 3.7, null, 3.0, 4.4, null, 3.4, 4.1, 3.6, 3.1, 4.0, 4.1, null, 3.6, 4.4, 2.9, 3.8, 4.0, 3.6, 3.6, 3.7, 3.8, 3.6, null, 3.5, null, 4.1, null, null, null, null, 4.3, 3.2, 3.3, 3.4, 3.9, 4.1, null, 3.0, null, null, 3.2, 3.4, 4.1, 4.0, null, 3.7, 3.3, 3.3, null, 4.0, null, 3.1, 3.2, 3.8, 3.2, 3.9, 4.3, 3.4, 3.7, 3.8, 3.2, 3.9, 3.2, null, 4.4, 3.3, 3.6, 3.3, null, 3.6, null, 3.9, null, 4.0, 3.8, 3.9, null, 3.8, null, 4.1, 4.2, 3.5, 3.8, null, null, 3.9, 3.6, 3.7, 4.5, 3.9, 3.2, 3.5, 3.3, 3.8, 3.1, 3.3, 2.3, 3.3, 3.7, null, 2.6, 3.4, 3.1, null, 3.6, 3.8, 4.0, 3.8, null, 3.2, null, null, 3.8, 3.6, 3.7, 3.6, 3.6, 4.0, 3.5, 3.8, 3.6, 3.6, 4.1, 4.1, null, 3.3, 3.1, null, 3.1, 3.6, 4.0, 3.4, 3.7, 3.9, 3.3, null, 3.6, 4.3, 3.2, 3.8, 3.7, 4.2, 3.3, 3.9, 3.8, 3.6, 3.6, 3.8, 4.1, 3.8, 4.0, null, 3.6, 3.9, null, 2.7, 4.4, 3.2, 3.8, 3.7, null, 2.6, null, 4.1, 3.7, 3.6, 3.5, null, null, 3.3, 3.0, 3.9, 3.6, 3.6, 3.2, null, 3.9, 3.5, 4.3, 3.2, null, 3.6, 3.2, 2.9, 3.6, 3.8, null, 3.2, 3.8, 4.1, null, 4.2, null, 3.3, 3.7, 3.6, null, null, 4.3, 3.6, 3.1, 3.9, 4.1, 4.1, 4.2, 3.9, 4.1, 4.0, 3.4, 4.5, null, 3.9, 3.4, 3.3, 4.1, 3.5, 4.0, null, 4.2, 2.6, null, 3.5, 3.5, 2.9, 3.8, 3.6, 3.7, 3.4, 4.6, 3.6, 3.8, 3.5, 4.3, 3.5, 3.1, 3.7, null, null, 4.3, 4.4, 3.9, 3.7, 4.1, 3.7, 3.5, 3.8, null, 3.0, null, 3.5, 3.0, 4.2, 3.2, 3.3, 3.2, 3.8, 3.5, 3.7, 4.5, 3.7, 2.9, 3.7, 3.6, 3.1, 3.2, null, null, 3.6, 3.5, 3.4, null, 4.1, 3.4, null, 3.3, 4.4, 3.4, null, 4.1, 3.4, 3.7, 4.3, 3.4, 3.3, 3.5, null, 3.5, 3.9, 4.0, 3.7, 4.4, 3.7, 3.3, 4.1, 2.8, 3.8, 3.9, 4.5, 3.3, null, 3.3, null, 3.7, 4.3, 3.9, 3.9, null, 3.1, null, 2.7, 3.7, 3.3, 4.3, 4.0, 3.2, null, 4.2, 4.3, 3.4, null, 3.8, null, null, 3.6, 4.3, 3.2, null, 3.5, 3.6, null, 2.8, 3.1, 2.9, 3.7, 3.6, null, 3.9, 3.5, 3.0, 3.8, 4.3, 3.4, 4.0, 3.9, 3.9, null, 3.6, 3.6, 4.2, null, null, 3.8, 3.0, 2.9, 4.3, 3.6, 3.7, 3.9, 4.5, 3.4, null, 4.2, 2.8, 4.2, null, 4.1, 3.0, 3.0, 4.6, 4.3, 3.3, 4.2, 3.6, null, 3.7, 3.0, 4.4, 4.1, 3.6, 4.2, 4.1, 3.3, 4.2, 4.1, null, 3.7, 3.8, 4.1, 3.9, 4.2, 4.1, 4.0, null, 3.6, 4.0, 3.5, 4.2, 3.9, 3.3, 4.0, 3.7, null, 3.3, 3.6, 3.9, 3.5, 3.6, 3.4, 3.4, 3.5, 3.3, 4.0, 3.7, 3.6, 4.3, 3.3, 2.8, 3.7, 3.8, 2.9, 4.2, 3.2, 4.0, null, 3.8, 3.9, null, 4.5, 3.9, 3.8, 3.3, 4.2, 4.2, 3.4, 3.9, 3.1, 3.8, 3.5, null, 3.0, 3.3, null, 3.7, 4.0, 3.6, 3.0, 3.6, null, 3.5, 4.0, null, 3.3, 3.7, 2.8, 3.4, 3.1, null, 4.0, 4.4, 3.8, 3.9, null, 2.8, 3.4, 3.8, 3.7, 4.2, 4.0, 3.8, 4.6, 3.1, 3.7, 4.0, 2.6, 3.5, 4.3, 3.9, 3.5, 3.5, 3.9, 3.5, null, 3.1, 4.2, 4.1, 3.8, 3.2, 3.3, 3.9, 3.4, 4.2, 3.7, 4.3, 2.9, 3.8, 3.8, 3.7, 4.1, 4.0, 4.3, 2.5, 4.3, 3.8, 3.1, 3.9, null, null, 3.5, 3.7, 2.9, 4.2, 4.4, 3.9, null, 3.9, 4.1, 3.4, null, 3.6, 3.9, 3.9, null, 3.2, 4.1, 3.2, 4.5, 3.5, 3.7, 3.1, 3.0, 4.3, null, 3.9, 4.3, 3.7, 3.9, 4.0, null, 3.9, 2.8, 4.0, 3.5, 4.0, 3.5, 3.4, 3.1, null, 3.1, 3.9, 4.3, 3.9, 3.8, 3.4, 4.2, null, 4.1, 4.2, 3.8, 3.1, 4.2, 3.7, 3.6, 3.0, 3.5, 3.4, 3.6, 3.6, 3.0, 4.2, 3.8, 2.2, 4.2, 3.5, 3.2, 4.1, 4.1, 2.9, 4.1, 3.8, 4.2, 3.8, 4.0, 3.8, 4.3, 3.8, null, 3.8, null, 3.7, null, null, 3.3, null, 3.8, 3.6, 2.9, 3.8, 3.4, 4.1, null, 4.0, 4.4, 3.9, 4.1, 4.4, 3.1, 3.6, null, 3.9, 3.1, 4.3, 3.5, 4.4, 2.9, 3.1, 4.1, 3.3, null, 3.8, null, 3.9, null, 3.8, 4.1, 4.1, 3.8, 4.0, 4.3, null, 3.2]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Book Table vs Rating"}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('141a8cb9-381b-425f-ae10-7c2e1a907801');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** Booking Table restaurant's are `more rated` than not booking table

##### Book Table vs Votes 


```python
plotly_box_plot(restaurant_df["book_table"], restaurant_df["no_of_votes"], 
                caption="Book Table vs Votes", color="Orange")
```


<div>


            <div id="bd765537-ff01-44e7-b8ad-b9c2c12e2c27" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("bd765537-ff01-44e7-b8ad-b9c2c12e2c27")) {
                    Plotly.newPlot(
                        'bd765537-ff01-44e7-b8ad-b9c2c12e2c27',
                        [{"marker": {"color": "Orange"}, "showlegend": false, "type": "box", "x": ["No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No"], "y": [0, 21, 131, 3236, 225, 402, 9, 712, 64, 46, 184, 0, 7, 13, 291, 89, 289, 0, 1214, 207, 433, 0, 280, 0, 121, 203, 0, 44, 0, 9, 214, 3468, 0, 114, 163, 94, 68, 263, 166, 208, 11, 7, 14, 1084, 0, 1258, 4, 99, 9, 0, 94, 960, 12, 19, 2487, 0, 100, 1026, 0, 47, 0, 2049, 1439, 10, 34, 237, 5, 604, 0, 18, 86, 833, 57, 0, 15, 370, 72, 75, 185, 89, 11, 96, 152, 894, 12, 171, 0, 0, 42, 23, 155, 22, 787, 74, 13, 0, 498, 0, 0, 14, 53, 0, 10, 49, 0, 782, 434, 0, 167, 29, 210, 21, 0, 51, 51, 18, 1426, 3870, 189, 11, 43, 103, 452, 165, 18, 29, 69, 0, 0, 0, 76, 46, 6, 241, 178, 181, 888, 82, 241, 0, 131, 0, 41, 40, 33, 9, 106, 11, 88, 2634, 719, 861, 89, 0, 0, 6, 285, 67, 3230, 0, 57, 0, 0, 0, 117, 135, 17, 503, 242, 18, 0, 484, 82, 49, 0, 0, 15, 0, 14, 8, 136, 0, 480, 21, 337, 0, 616, 218, 269, 0, 110, 118, 375, 23, 31, 142, 79, 21, 334, 476, 5, 1154, 0, 1225, 39, 130, 27, 0, 101, 32, 132, 7330, 0, 0, 131, 9, 1808, 53, 0, 0, 28, 0, 410, 0, 0, 28, 32, 324, 347, 201, 20, 34, 327, 41, 592, 791, 154, 822, 195, 0, 942, 1847, 0, 23, 163, 289, 12, 16, 179, 235, 59, 0, 60, 0, 597, 240, 0, 47, 59, 0, 8, 21, 9, 12, 994, 994, 421, 12, 0, 30, 11, 0, 238, 36, 47, 759, 27, 17, 16, 0, 14, 57, 21, 168, 420, 217, 253, 1708, 120, 97, 0, 0, 33, 12, 53, 1048, 108, 18, 450, 0, 0, 19, 0, 47, 11, 104, 38, 11, 355, 232, 632, 23, 48, 77, 189, 638, 38, 111, 41, 8, 183, 5, 8, 5, 85, 523, 27, 291, 402, 0, 14, 93, 233, 97, 800, 39, 2291, 0, 7, 27, 9, 41, 59, 138, 548, 23, 151, 0, 44, 24, 0, 73, 10, 10, 48, 11, 552, 38, 27, 131, 971, 0, 289, 494, 0, 176, 74, 181, 49, 229, 17, 519, 76, 1053, 152, 0, 11, 185, 1241, 1269, 0, 5, 1543, 889, 0, 0, 17, 1289, 13, 559, 49, 66, 0, 0, 100, 178, 272, 10, 0, 0, 427, 8, 504, 15, 5, 0, 487, 0, 42, 67, 89, 148, 324, 34, 170, 11, 0, 9, 0, 9, 0, 66, 138, 38, 4, 253, 11, 4, 0, 0, 8, 212, 0, 58, 110, 24, 71, 1229, 294, 136, 790, 129, 57, 185, 21, 27, 4, 35, 60, 50, 7, 16, 0, 553, 570, 89, 100, 196, 0, 132, 1461, 7, 261, 13, 12, 197, 0, 0, 0, 10, 759, 18, 270, 13, 278, 282, 28, 290, 86, 385, 0, 57, 4, 507, 399, 0, 118, 56, 0, 531, 108, 0, 32, 621, 20, 0, 13, 109, 9, 17, 3621, 0, 450, 25, 1077, 776, 0, 64, 20, 837, 110, 6, 403, 91, 155, 1792, 128, 0, 66, 7854, 23, 1087, 0, 24, 0, 499, 50, 448, 226, 0, 9, 5, 0, 0, 53, 70, 0, 10, 14, 0, 57, 6, 0, 88, 73, 0, 3592, 0, 429, 41, 41, 345, 0, 91, 479, 1703, 38, 0, 972, 559, 85, 10, 35, 571, 728, 33, 46, 63, 33, 0, 37, 0, 11, 233, 0, 67, 444, 2055, 0, 4, 7, 605, 55, 0, 476, 4, 0, 6, 28, 0, 0, 100, 0, 12, 1095, 0, 0, 3236, 5, 8, 48, 48, 94, 4, 243, 59, 17, 42, 436, 0, 80, 19, 43, 165, 0, 164, 2332, 125, 676, 26, 448, 247, 11, 14, 92, 0, 1345, 727, 115, 0, 362, 0, 4, 701, 364, 446, 48, 287, 1750, 435, 83, 101, 46, 182, 290, 508, 64, 28, 48, 433, 16, 45, 31, 110, 57, 6, 4, 47, 0, 0, 6, 199, 0, 512, 392, 7, 9, 33, 7, 0, 7, 147, 783, 53, 24, 885, 19, 19, 0, 53, 68, 0, 20, 104, 59, 31, 289, 24, 199, 189, 539, 58, 34, 0, 290, 45, 553, 6, 96, 187, 61, 2164, 11, 0, 27, 14, 69, 56, 13, 914, 110, 4, 19, 192, 148, 16, 1214, 4, 0, 113, 0, 182, 26, 225, 23, 92, 29, 4, 156, 10, 0, 28, 53, 53, 0, 276, 0, 50, 71, 0, 180, 48, 0, 10, 151, 0, 3468, 25, 32, 229, 6, 25, 281, 468, 29, 18, 3238, 0, 0, 3163, 9, 22, 0, 0, 4, 8, 185, 17, 18, 0, 800, 130, 21, 1049, 120, 19, 145, 109, 20, 283, 1310, 0, 56, 220, 539, 1142, 747, 152, 106, 52, 8, 0, 635, 36, 18, 254, 61, 157, 61, 9, 152, 168, 12, 34, 420, 37, 514, 8, 0, 4, 0, 136, 58, 15, 13, 0, 294, 251, 0, 17, 6, 137, 5, 984, 718, 122, 725, 0, 51, 19, 13, 88, 242, 479, 51, 4, 463, 0, 39, 0, 13, 106, 420, 140, 10, 4, 197, 0, 326, 0, 786, 68, 22, 146, 19, 0, 462, 7, 203, 53, 60, 171, 96, 142, 42, 0, 1320, 0, 76, 153, 442, 6, 0, 2662, 10, 200, 0, 4, 8, 0, 7, 34, 11, 76, 0, 337, 42, 0, 0, 0, 5, 78, 510, 399, 283, 118, 36, 62, 48, 21, 549, 0, 0, 266, 5, 168, 15, 3486, 18, 31, 520, 16, 7, 19, 71, 19, 84, 12, 2714, 0, 1156, 125, 0, 1177, 66, 3712, 37, 0, 0, 96, 33, 16, 113, 92, 1972, 27, 42, 1187, 1359, 69, 57, 159, 10, 32, 0, 55, 24, 0, 55, 56, 44, 61, 0, 783, 406, 126, 0, 1745, 24, 4, 21, 34, 94, 591, 70, 11, 46, 0, 0, 13, 253, 0, 0, 620, 5, 7, 29, 32, 10, 236, 0, 16, 5, 195, 19, 17, 0, 332, 0, 984, 0, 0, 2508, 47, 4, 1370, 819, 0, 4, 10, 72, 25, 14, 38, 7, 1175, 0, 13, 4884, 500, 25, 0, 23, 362, 0, 331, 0, 536, 251, 0, 0, 31, 70, 1647, 250, 0, 225, 87, 65, 0, 558, 0, 11, 29, 100, 0, 47, 178, 0, 22, 2073, 0, 9, 5, 25, 0, 0, 70, 102, 951, 67, 568, 485, 9, 241, 456, 277, 464, 89, 2450, 0, 45, 166, 0, 273, 1106, 623, 220, 229, 155, 0, 38, 0, 2389, 19, 0, 11, 1871, 500, 4, 0, 257, 48, 0, 30, 949, 7, 6, 113, 111, 16, 0, 100, 4694, 476, 7, 21, 386, 0, 1858, 109, 36, 50, 4, 34, 2720, 60, 0, 177, 17, 110, 0, 383, 372, 11, 17, 9, 4, 3624, 130, 0, 0, 0, 184, 0, 1187, 357, 511, 12, 450, 158, 12, 9, 133, 0, 76, 0, 111, 26, 79, 35, 1856, 15, 790, 15, 1878, 0, 0, 34, 680, 7, 0, 191, 12, 0, 39, 46, 63, 0, 111, 83, 44, 130, 221, 3236, 1804, 21, 0, 79, 13, 147, 11, 6, 488, 30, 410, 0, 970, 0, 19, 236, 18, 2852, 137, 434, 10, 0, 176, 0, 0, 1858, 73, 34, 34, 4, 289, 0, 0, 34, 819, 109, 14, 18, 505, 9, 12, 17, 0, 0, 19, 115, 123, 0, 360, 0, 468, 180, 34, 47, 311, 13, 14, 924, 16, 7, 863, 19, 191, 0, 1223, 118, 2867, 261, 1431, 211, 7, 323, 302, 25, 37, 14, 54, 194, 860, 3843, 159, 0, 0, 0, 28, 0, 7, 6, 0, 64, 995, 63, 0, 31, 8, 776, 417, 16, 47, 6, 149, 59, 15, 2198, 16, 866, 122, 2041, 4, 82, 126, 192, 5, 0, 8, 0, 419, 5, 0, 789, 14, 2032, 10, 38, 0, 39, 2332, 0, 7, 98, 0, 0, 0, 0, 167, 1508, 432, 185, 9, 0, 0, 0, 16, 4, 4, 185, 180, 0, 175, 76, 79, 0, 16, 215, 0, 208, 69, 67, 790, 211, 38, 39, 3991, 81, 4, 0, 39, 197, 74, 22, 166, 2577, 233, 4811, 0, 213, 835, 132, 267, 853, 4, 195, 0, 0, 109, 0, 6, 1203, 42, 0, 24, 94, 6, 11, 6, 1238, 0, 0, 1785, 23, 275, 20, 0, 0, 108, 337, 92, 24, 117, 70, 169, 129, 0, 376, 61, 33, 72, 17, 4, 100, 681, 0, 82, 166, 22, 0, 8, 2304, 0, 1196, 70, 247, 0, 21, 140, 0, 11, 0, 41, 151, 2447, 4, 12, 0, 0, 20, 7, 38, 21, 142, 54, 783, 31, 239, 44, 9, 154, 610, 109, 9, 1920, 15, 57, 50, 7, 245, 0, 19, 215, 84, 0, 4, 22, 0, 14, 0, 11, 679, 0, 9, 1324, 198, 330, 25, 894, 0, 45, 864, 57, 38, 127, 25, 539, 30, 257, 11, 0, 31, 0, 845, 0, 0, 0, 0, 1332, 23, 4, 17, 150, 161, 0, 15, 0, 0, 11, 7, 271, 153, 0, 25, 47, 111, 0, 160, 0, 12, 68, 69, 4, 191, 570, 99, 26, 158, 4, 88, 4, 0, 1865, 4, 23, 82, 0, 42, 0, 370, 0, 76, 194, 599, 0, 232, 0, 1850, 470, 9, 27, 0, 0, 42, 112, 25, 3987, 125, 10, 50, 5, 71, 5, 12, 235, 5, 47, 0, 84, 6, 25, 0, 156, 258, 427, 148, 0, 5, 0, 0, 214, 136, 126, 12, 55, 958, 15, 94, 15, 23, 205, 201, 0, 4, 28, 0, 11, 13, 154, 5, 24, 49, 218, 0, 17, 361, 7, 37, 66, 164, 395, 60, 41, 15, 23, 31, 1503, 187, 101, 0, 24, 240, 0, 29, 63, 15, 75, 82, 0, 30, 0, 145, 21, 221, 13, 0, 0, 4, 11, 125, 17, 68, 6, 0, 122, 29, 1052, 4, 0, 28, 17, 17, 5, 63, 0, 16, 32, 520, 0, 957, 0, 11, 25, 61, 0, 0, 1721, 92, 44, 65, 237, 563, 515, 358, 600, 124, 5, 188, 0, 228, 946, 5, 785, 21, 284, 0, 289, 74, 0, 25, 15, 19, 90, 39, 382, 7, 4694, 52, 31, 21, 429, 16, 22, 8, 0, 0, 754, 69, 190, 25, 2773, 14, 11, 30, 0, 21, 0, 15, 34, 176, 4, 23, 9, 46, 43, 145, 1013, 16, 45, 259, 9, 6, 4, 0, 0, 70, 54, 9, 0, 2461, 54, 0, 4, 4460, 4, 0, 292, 15, 37, 454, 5, 39, 89, 0, 6, 652, 42, 25, 2182, 127, 5, 399, 161, 110, 224, 194, 42, 0, 12, 0, 41, 1135, 823, 135, 0, 7, 0, 31, 11, 9, 2230, 578, 6, 0, 226, 62, 257, 0, 117, 0, 0, 16, 286, 5, 0, 19, 14, 0, 511, 150, 21, 63, 68, 0, 43, 12, 36, 377, 654, 12, 339, 66, 59, 0, 23, 11, 570, 0, 0, 94, 31, 243, 3126, 32, 195, 218, 3471, 8, 0, 1428, 270, 115, 0, 2317, 110, 39, 206, 687, 13, 1476, 97, 0, 28, 25, 2861, 221, 112, 2034, 295, 16, 980, 847, 0, 16, 191, 573, 501, 744, 724, 409, 131, 19, 932, 20, 215, 134, 47, 1310, 49, 0, 4, 123, 42, 56, 16, 39, 225, 13, 4, 118, 329, 22, 1143, 7, 102, 35, 151, 77, 624, 5, 28, 0, 23, 331, 0, 804, 33, 85, 6, 2049, 713, 20, 1190, 43, 50, 12, 0, 10, 24, 0, 181, 632, 383, 404, 14, 0, 14, 130, 0, 15, 270, 36, 11, 5, 0, 749, 4079, 23, 131, 0, 185, 17, 38, 17, 240, 314, 111, 2490, 6, 15, 152, 408, 8, 2139, 169, 11, 11, 118, 28, 0, 121, 386, 568, 27, 4, 8, 347, 8, 4733, 30, 7544, 24, 200, 56, 31, 362, 76, 1048, 23, 379, 41, 10, 38, 0, 0, 14, 27, 26, 176, 7210, 888, 0, 66, 164, 7, 0, 39, 1791, 2322, 0, 17, 313, 21, 3991, 33, 26, 38, 6, 548, 0, 727, 1559, 30, 54, 856, 0, 237, 58, 40, 10, 60, 75, 48, 104, 0, 105, 592, 1593, 32, 63, 23, 583, 0, 60, 876, 376, 23, 174, 51, 24, 16, 140, 8, 8, 108, 9, 53, 91, 409, 1901, 6, 6, 159, 493, 93, 184, 149, 858, 69, 34, 706, 324, 377, 0, 122, 0, 21, 0, 0, 11, 0, 101, 68, 18, 410, 422, 176, 0, 89, 568, 61, 262, 751, 253, 112, 0, 851, 13, 61, 4, 1227, 20, 68, 988, 11, 0, 273, 0, 96, 0, 693, 341, 95, 214, 1013, 2039, 0, 4]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Book Table vs Votes"}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('bd765537-ff01-44e7-b8ad-b9c2c12e2c27');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** Booking Table restaurant's are `more voted` than not booking table

##### Book Table vs Costs


```python
plotly_box_plot(restaurant_df["book_table"], restaurant_df["cost_for_two_people"], 
                caption="Book Table vs Costs for Two People", color="Red")
```


<div>


            <div id="89f69a1e-931b-4dd8-8367-2210d704f42d" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("89f69a1e-931b-4dd8-8367-2210d704f42d")) {
                    Plotly.newPlot(
                        '89f69a1e-931b-4dd8-8367-2210d704f42d',
                        [{"marker": {"color": "Red"}, "showlegend": false, "type": "box", "x": ["No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes", "No", "No"], "y": ["200", "200", "230", null, "800", "1,200", "250", "1,000", "250", "450", "350", "300", "200", "250", "500", "350", "800", "300", "1,200", "1,000", "800", "400", "1,500", "150", "600", "900", "400", "100", "400", "300", "600", "1,300", "300", "1,500", "500", "300", "450", "700", "400", "400", "200", "400", "500", "600", "200", "1,000", "600", "200", "400", "300", "500", "600", "200", "700", "900", "150", "300", "800", "300", "200", "400", "1,600", "2,200", "100", "200", null, "1,000", "1,800", "300", "400", "500", "700", "200", "250", "100", "1,000", "300", "500", "600", "400", "300", "300", "700", "750", "700", "400", "400", "300", "800", "300", "400", "400", "650", "600", "400", "400", "2,100", "300", "350", "250", "300", "100", "250", "300", "150", "1,500", "800", "800", "1,700", "400", "800", "200", "150", "300", "500", "600", "1,700", "1,600", "400", "50", "600", "700", "600", "3,000", "500", "300", "700", "150", "600", "300", "400", "350", "200", "500", "800", "250", "650", "200", "500", "550", "700", "300", "700", "300", "600", "300", "600", "200", "500", "500", "950", "1,800", "200", "300", "100", "300", "800", "600", "1,600", "450", "400", "150", "300", "350", "2,000", "800", "800", "700", "400", "200", "700", "3,000", "200", "200", "300", "1,500", "400", "200", "200", "150", "600", "400", "450", "200", "1,000", "200", "500", "400", "300", "300", "200", "1,000", "3,000", "500", "500", "550", "500", "700", "550", "1,400", "400", "1,100", "600", "400", "950", "300", "600", "900", "250", "700", "300", "1,400", "150", "300", "400", "550", "1,000", "750", "600", "1,500", "200", "300", "500", "1,200", "200", "500", "200", "700", "1,000", "400", "400", "200", "500", "100", "1,300", "1,000", "200", "1,000", "3,000", "200", "1,300", "600", "250", "500", "500", "400", "200", "250", "300", "1,000", "300", "200", "400", "200", "2,500", "600", "150", "600", "200", "150", "600", "150", "300", "150", "1,300", "1,300", "800", "200", "200", "800", "250", "300", "800", "150", "200", "1,200", "550", "400", "1,100", "300", "200", "350", "250", "2,500", "700", "1,500", "1,000", "1,300", "300", "2,400", "500", "300", "500", "150", "350", "1,600", "450", "100", "400", "250", "100", "400", "200", "100", "200", "1,200", "900", "200", "500", "1,000", "500", "350", "250", "250", "300", "400", "350", "800", "200", "250", "550", "200", "600", "600", "600", "2,500", "300", "750", "1,000", "500", "300", "200", "500", "650", "750", "700", "1,400", "300", "300", "1,000", "700", "400", "300", "500", "1,200", "800", "1,350", "400", "500", "550", "250", "250", "100", "500", "1,500", "1,000", "800", "400", "200", "800", "1,500", "500", "400", "800", "100", "400", "400", "550", "400", "400", "300", "400", "500", "500", "300", "500", "300", "300", "1,500", "1,000", "400", "100", "1,200", "800", "250", "200", "400", "1,100", "200", "1,100", "300", "800", "250", "650", "1,900", "700", "600", "1,000", "250", "500", "500", "200", "350", "350", "450", "200", "700", "200", "400", "600", "1,400", "600", "180", "150", "1,300", "200", "350", "300", "300", "250", "300", "800", "800", "900", "200", "1,000", "250", "300", "400", "500", "300", "400", "200", "300", "200", "500", "1,000", "700", "600", "1,500", "1,000", "150", "250", "500", "300", "300", "200", "100", "400", "400", "100", "200", "400", "900", "1,200", "300", "150", "600", "400", "750", "1,500", "200", "500", "500", "500", "1,000", "300", null, "150", "250", "500", "500", "4,000", "350", "200", "500", "350", "450", "600", "1,500", "200", "400", "250", "1,000", "1,000", "500", "500", "500", "500", "800", "700", "300", "500", "3,000", "550", "500", "600", "300", "150", "200", "400", "300", "700", "450", "400", "1,000", "200", "400", "500", "1,100", "500", "500", "400", "550", "300", "1,200", "600", "250", "150", "1,800", "300", "550", "100", "550", "400", "850", "500", "700", "200", "200", "300", "400", "500", "300", "600", "250", "400", "500", "300", "300", "500", "400", "300", "500", "600", "500", "2,100", "500", "3,400", "700", "600", "400", "300", "600", "650", "650", "400", "1,000", "750", "500", "400", "500", "350", "4,000", "800", "400", "200", "600", "300", "1,500", "400", "150", "700", "300", "100", "500", "100", "1,200", "300", "200", "400", "800", "300", "500", "1,700", "500", "200", "400", "250", "200", "350", "2,400", "400", "300", "1,500", "500", "500", null, "350", "250", "350", "150", "250", "350", "500", "300", "400", "2,000", "500", "200", "800", "600", "150", "2,800", "300", "400", "700", "900", "650", "300", "600", "650", "300", "150", "400", "200", "1,700", "1,500", "550", "550", "1,000", "400", "150", "600", "550", "2,000", "300", "150", "1,000", "500", "400", "200", "300", "450", "500", "1,000", "250", "2,000", "3,000", "500", "500", "550", "200", "150", "1,000", "200", "200", "500", "400", "150", "600", "500", "200", "400", "500", "200", "300", "400", "250", "300", "300", "400", "750", "600", "300", "900", "700", "300", "400", "400", "550", "350", "700", "700", "300", "400", "850", "500", "500", "400", "500", "250", "350", "250", "400", "400", "900", "300", "1,100", "500", "250", "1,000", "500", "400", "400", "600", "650", "200", "700", "1,000", "300", "400", "400", "1,000", "750", "150", "1,200", "100", "300", "400", "150", "600", "400", "2,000", "350", "600", "2,200", "450", "400", "250", "600", "200", "400", "600", "500", "250", "300", "550", "400", "400", "250", "650", "300", "300", "450", "400", "600", "500", "350", "350", "450", "150", "500", "700", "100", "200", "800", "200", "80", "1,500", "300", "500", "200", "700", "400", "800", "800", "500", "200", "200", "750", "400", "250", "600", "600", "400", "600", "120", "200", "350", "750", "150", "500", "500", "500", "500", "1,500", "600", "600", "300", "250", "300", "400", "300", "400", "1,050", "300", "450", "500", "250", "1,500", "600", "250", "600", "800", "400", "600", "300", "150", "250", "200", "1,200", "900", "400", "400", "300", "850", "200", "400", "200", "300", "300", "400", "1,000", "600", "700", "1,700", "250", "200", "400", "300", "500", "400", "650", "300", "400", "800", "450", "400", "400", "400", "450", "700", "600", "500", "300", "300", "250", "500", "800", "1,500", "1,000", "150", "600", "500", "300", "600", "1,000", "800", "350", "400", "500", "1,100", "500", "400", "150", "1,200", "550", "300", "800", "900", "250", "300", "600", "100", "300", null, "350", "200", "400", "200", "350", "800", "450", "200", "200", "150", "300", "250", "200", "300", "450", "700", "700", "600", "800", "200", "400", "1,500", "700", "1,000", "400", "1,000", "1,000", "400", "500", "350", "1,300", "300", "300", "300", "300", "350", "1,100", "500", "200", "300", "500", "1,000", "400", "500", "600", "150", "1,300", "200", "1,300", "400", "250", "300", "1,300", "100", "150", "1,000", "700", "800", "450", "400", "550", "700", "700", "400", "300", "150", "350", "300", "400", "500", "300", "600", "1,200", "200", "1,000", "450", "750", "1,000", "300", "300", "1,400", "600", "200", "300", "500", "300", "700", "650", "200", "2,000", "400", "400", "300", "300", "400", "100", "3,400", "200", "200", "800", "200", "300", "1,250", "400", "150", "400", "500", "200", "300", "400", "300", "300", "1,000", "800", "250", "900", "250", "400", "1,000", "800", "1,500", "300", "1,300", "2,500", "350", "400", "500", "150", "1,300", "450", "300", "750", null, "750", "800", "450", "600", "300", "200", "200", "450", "350", "300", "500", "800", "600", "600", "500", "300", "1,200", "200", "350", "300", "150", "200", "300", "150", "500", "300", "150", "300", "400", "650", "1,200", "500", "200", "150", "300", "200", "200", "400", "300", "1,100", "500", "1,000", "700", "250", "100", "1,500", "2,000", "300", "350", "700", "400", "300", "700", "200", "900", "600", "3,000", "400", "500", "500", "600", "500", "1,000", "1,000", "300", "150", "300", "600", "600", "100", "150", "450", "400", "500", "500", "900", "200", "300", "200", "800", "100", "150", "700", "1,500", "600", "450", "850", "700", "250", "1,200", "1,500", "300", "550", "300", "300", "1,200", "900", "400", "500", "550", "500", "200", "700", "500", "400", "100", "300", "450", "400", "400", "200", "500", "300", "300", "250", "550", "500", "500", "300", "400", "300", "600", "600", "1,500", "300", "300", "150", "700", "300", "300", "200", "1,100", "400", "400", "400", "900", "300", "250", "400", "1,200", "500", "200", "1,000", "300", "200", "200", "300", "300", null, "1,400", "800", "500", "300", "400", null, "1,000", "600", "250", "1,200", "700", "400", "250", "300", "1,900", "500", "300", "300", "1,500", "200", "350", "750", "300", "1,100", "300", "400", "250", "200", "450", "400", "600", "1,200", "250", "500", "500", "200", "500", "400", "200", "300", "300", "500", "550", "450", "500", "300", "700", "300", "400", "300", "200", "700", "1,100", "300", "400", "200", "200", "800", "300", "350", "800", "150", "300", "300", "200", "250", "800", "400", "300", "400", "1,400", "1,500", "750", "700", "800", "300", "500", "850", "500", "200", "200", "300", "250", "800", "1,800", "800", "350", "600", "150", "500", "500", "500", "300", "400", "400", "250", "350", "600", "300", "150", "450", "800", "1,000", "500", "800", "350", "800", "300", "150", "1,800", "400", "1,400", "1,200", "1,200", "200", "400", "750", "450", "150", "300", "500", "800", "1,400", "100", "400", "400", "350", "1,000", "400", "500", "150", "600", "2,800", "200", "500", "800", "450", "800", "150", "500", "400", "1,900", "500", "800", "200", "500", "150", "300", "150", "250", "300", "400", "1,300", "400", "800", "300", "400", "200", "600", "1,000", "500", "650", "700", "150", "1,300", "400", "200", "550", "1,600", "150", "600", "200", "500", "700", "600", "400", "250", "1,200", "500", "1,300", "800", null, "450", "250", "200", "1,400", "1,200", "400", "300", "200", "450", "500", "150", "2,000", "300", "200", "850", "1,100", "300", "150", "150", "1,700", "350", "400", "1,200", "150", "4,000", "500", "1,000", "150", "1,000", "300", "350", "250", "200", "750", "1,100", "800", "200", "350", "600", "400", "300", "400", "200", "800", "500", "400", "300", "600", "500", "200", "250", "800", "400", "1,500", "800", "450", "150", "600", "300", "250", "200", "350", "150", "500", "800", "300", "300", "250", "300", "300", "1,000", "300", "200", "400", "400", "300", "250", "600", "600", "150", "400", "1,000", "1,800", "1,000", "1,500", "300", "800", "550", "150", "300", "400", "300", "600", "200", "400", "600", "250", "600", "250", "200", "400", "1,500", "400", "400", "800", "500", "500", "1,100", "800", "200", "700", "3,000", "600", "300", "900", "1,600", "800", "400", "600", "150", "150", "300", "300", "700", "100", "500", "250", "300", "1,000", "400", "300", "400", "500", "300", "400", "600", "300", "700", "200", "250", "500", "700", "200", "200", "300", "300", "300", "700", "200", "800", "400", "700", "400", "300", "700", "400", "200", "1,300", "200", "1,000", "300", "250", "1,800", "400", "350", "500", "150", "200", "400", "200", "200", "300", "800", "1,300", "400", "500", "400", "600", "400", "200", "1,200", "200", "300", "400", "500", "400", "1,600", "500", "500", "600", "300", "300", "400", "300", "800", "400", "250", "850", "400", "300", "150", "150", "300", "300", "2,400", "150", "500", "250", "300", "1,000", "700", "600", "600", "500", "200", "500", "400", "250", "700", "450", "250", "300", null, "100", "200", "250", "300", "200", "500", "150", "400", "600", "550", "500", "200", "3,500", "200", "200", "800", "750", "400", "300", "400", "300", "300", "600", "1,500", "500", "150", "300", "300", "400", "300", "600", "300", "250", "1,000", "300", "300", "500", "150", "200", "150", "600", "700", "400", "150", "1,400", "400", "200", "800", "300", "100", "150", "250", "2,200", "1,600", "600", "400", "600", "550", "150", "300", "350", "250", "350", "350", "300", "800", "800", "300", "600", "500", "250", "250", "400", "1,300", "350", "150", "100", "1,000", "350", "800", "400", "400", "200", "350", "1,200", "200", "600", null, "400", "1,500", "500", "1,500", "150", "400", "700", "200", "150", "400", "450", "650", "600", "600", "300", "1,500", "600", "150", "450", "3,400", "600", "400", "100", "1,100", "600", "500", "300", "600", "400", "1,500", "200", "400", "600", "500", "600", "400", "300", "500", "800", "300", "400", "1,800", "450", "450", "1,000", "1,100", "600", "600", "750", "300", "200", "350", "350", "400", "600", "1,800", "800", "400", "1,000", "250", "400", "350", "150", "600", "100", "600", "150", "600", "3,000", "300", "250", "500", "400", "400", "500", "300", "300", "1,400", "700", "300", "300", "1,200", "1,400", "250", "200", "500", "250", "250", "200", "250", "800", "550", "800", "600", "300", "200", "500", "1,200", "250", "1,700", "600", "400", "500", "1,000", "250", "250", "200", "350", "80", "300", "200", "150", "550", "350", "1,000", "200", "300", "350", "750", "600", "300", "550", "200", "200", "300", "200", "500", "750", "300", "550", "200", "500", "100", "500", "700", "1,200", "350", "300", "250", "150", "300", "1,500", "400", "400", "600", "1,300", "550", "300", "1,300", "500", "400", "500", "950", "650", "400", "200", "3,000", "600", "1,500", "300", "150", "500", "300", "1,500", "1,000", "500", "1,200", "700", "400", "1,000", "1,100", "300", "250", "300", "600", "300", "750", "300", "350", "650", "350", "1,000", "550", "1,500", "550", "300", "750", "300", "250", "500", "600", "500", "1,800", "150", "500", "300", "400", "200", "250", "500", "350", "1,100", "350", "350", "350", "400", "400", "300", "250", "500", "300", null, "400", "500", "800", "300", "300", "300", "1,000", "500", "1,200", "500", "500", "250", "650", "500", "300", "300", "200", "500", "800", "600", "500", "200", "200", "250", "230", "400", "900", "500", "550", "300", "400", "250", "1,100", "1,200", null, "450", "500", "500", "400", "400", "200", "1,500", "800", "600", "1,600", "300", "200", "350", "1,000", "600", "600", "600", "250", "300", "800", "350", "250", "500", "500", "2,500", "150", "500", "500", "600", "400", "1,500", "350", "1,500", "250", "1,600", "400", "350", "400", "300", "1,600", "600", "700", "500", "900", "400", "200", "250", "300", "700", "300", null, "600", "750", "200", "200", "600", "300", "350", "400", "1,300", "700", "250", "500", "650", "600", "1,600", "600", "700", "300", "200", "1,000", "300", "800", "1,200", "900", "1,500", "650", "250", null, "600", "300", "800", "400", "500", "550", "250", "250", "400", "1,300", "550", "800", "300", "400", "700", "350", "600", "600", "600", "400", "600", "300", "250", "150", "600", "150", "400", "450", "450", "400", "1,800", "1,000", "800", "400", "250", "300", "600", "400", "550", "250", "1,700", "300", "200", "750", "180", "500", "400", null, "200", "500", "350", "150", "300", "300", "350", "300", "450", "350", "400", "800", "300", "2,500", "400", "200", "700", "3,000", "300", "300", "300", "700", "400", "700", "200", "800", "600", "550", "600", "400", "200", "900", "500", "350", "400", "750", "600", "600", "700", "1,200", "1,700", "300", "150"]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Book Table vs Costs for Two People"}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('89f69a1e-931b-4dd8-8367-2210d704f42d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** Booking Table restaurant's are `more costly` than not booking table

##### Rating & Votes & Costs Distribution


```python
plotly_histogram_chart(restaurant_df["rating"], x_title="Rating", color="Green")
```


<div>


            <div id="d7eb2050-02fc-4560-8c22-fbe2066367b5" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("d7eb2050-02fc-4560-8c22-fbe2066367b5")) {
                    Plotly.newPlot(
                        'd7eb2050-02fc-4560-8c22-fbe2066367b5',
                        [{"marker": {"color": "Green"}, "opacity": 0.75, "type": "histogram", "x": [null, 3.1, 4.0, 4.2, 3.9, 4.1, 3.3, 4.4, 3.9, 3.8, 3.3, null, 3.1, 3.5, 3.8, 2.9, 4.1, null, 3.9, 4.1, 3.8, null, 4.4, null, 4.1, 3.7, null, 3.5, null, 3.4, 3.7, 4.5, null, 4.3, 3.6, 2.8, 3.6, 3.8, 4.0, 3.8, 3.3, 3.4, 3.5, 4.4, null, 4.3, 3.2, 4.0, 2.9, null, 3.1, 4.0, 3.0, 3.4, 4.4, null, 3.9, 4.4, null, 3.2, null, 4.1, 4.6, 3.6, 3.6, 3.9, 3.4, 4.1, null, 3.5, 3.7, 3.7, 3.9, null, 3.9, 4.0, 3.8, 2.9, 3.9, 3.8, 2.8, 3.6, 2.8, 4.0, 3.7, 3.8, null, null, 3.1, 2.9, 3.9, 3.2, 4.0, 3.7, 3.0, null, 4.8, null, null, 3.6, 3.6, null, 3.3, 3.7, null, 4.1, 3.8, null, 4.0, 3.8, 4.0, 3.5, null, 4.0, 3.6, 2.9, 4.5, 4.4, 2.8, 3.6, 3.8, 4.4, 4.1, 4.3, 3.8, 3.7, 3.7, null, null, null, 2.9, 3.4, 3.5, 3.0, 3.8, 3.9, 3.5, 3.1, 3.0, null, 3.8, null, 3.1, 4.0, 3.8, 3.4, 3.7, 3.4, 3.0, 4.4, 4.1, 4.4, 4.0, null, null, 3.3, 3.8, 2.9, 4.2, null, 3.0, null, null, null, 4.1, 3.9, 3.6, 3.8, 2.1, 3.6, null, 4.2, 3.8, 3.9, null, null, 3.5, null, 3.2, 3.5, 3.6, null, 4.4, 3.6, 4.2, null, 4.1, 3.6, 4.0, null, 3.8, 4.1, 4.4, 3.7, 3.1, 4.0, 3.7, 3.7, 3.4, 4.3, 3.3, 4.4, null, 4.1, 3.7, 3.4, 3.6, null, 4.1, 3.6, 3.8, 4.2, null, null, 4.1, 3.1, 4.4, 3.1, null, null, 3.9, null, 3.9, null, null, 3.0, 3.8, 4.0, 4.0, 4.9, 3.0, 4.0, 3.9, 3.8, 4.0, 4.3, 3.9, 4.2, 4.0, null, 4.0, 4.1, null, 3.7, 4.1, 4.2, 3.5, 3.3, 3.9, 3.9, 3.9, null, 3.1, null, 4.3, 4.2, null, 3.7, 4.1, null, 3.3, 3.0, 3.2, 3.4, 4.3, 4.3, 3.9, 3.6, null, 4.0, 3.5, null, 3.9, 3.4, 3.7, 4.1, 3.4, 3.6, 3.1, null, 3.4, 3.9, 3.7, 4.1, 3.7, 4.2, 2.8, 4.3, 3.8, 4.0, null, null, 3.7, 3.5, 3.7, 4.3, 3.6, 4.0, 4.3, null, null, 3.5, null, 3.8, 3.3, 4.0, 3.8, 3.5, 3.9, 4.1, 3.8, 3.4, 3.9, 3.8, 3.8, 4.4, 3.2, 3.7, 3.8, 3.6, 4.1, 3.3, 3.5, 3.3, 3.7, 4.4, 3.6, 3.7, 2.7, null, 3.5, 3.7, 4.0, 3.7, 4.2, 3.7, 4.2, null, 3.5, 3.6, 3.5, 4.1, 4.0, 4.0, 4.3, 3.8, 3.8, null, 3.8, 3.7, null, 3.7, 3.5, 3.6, 4.0, 2.8, 4.1, 3.7, 3.9, 3.5, 4.1, null, 4.2, 4.0, null, 3.7, 4.3, 4.0, 4.0, 3.7, 3.5, 3.6, 4.0, 4.4, 3.9, null, 3.2, 4.2, 4.1, 4.3, null, 3.3, 4.1, 4.1, null, null, 3.5, 4.2, 3.6, 4.1, 3.9, 4.0, null, null, 3.9, 2.8, 4.1, 3.5, null, null, 3.8, 3.5, 4.5, 3.6, 3.2, null, 3.9, null, 3.5, 3.3, 3.9, 3.0, 4.3, 3.9, 3.8, 3.5, null, 3.4, null, 3.5, null, 3.7, 4.1, 3.8, 3.2, 2.8, 3.5, 3.3, null, null, 3.4, 4.0, null, 3.5, 3.9, 3.7, 3.8, 4.0, 3.2, 4.2, 4.3, 3.7, 4.0, 2.8, 3.3, 3.4, 3.3, 3.7, 3.8, 3.7, 3.3, 3.7, null, 4.1, 4.2, 4.0, 4.0, 3.6, null, 3.7, 4.2, 3.2, 3.8, 3.5, 3.3, 3.8, null, null, null, 3.4, 3.8, 3.8, 4.1, 3.2, 3.8, 4.0, 3.7, 3.0, 3.2, 4.4, null, 3.3, 3.2, 4.0, 3.7, null, 4.1, 2.6, null, 4.1, 4.1, null, 3.6, 3.8, 2.8, null, 2.9, 4.0, 3.0, 3.7, 4.3, null, 3.9, 3.5, 4.2, 4.0, null, 3.6, 3.5, 3.7, 3.8, 3.4, 4.2, 3.6, 4.5, 4.1, 3.7, 3.9, 3.7, 4.5, 3.6, 4.0, null, 3.7, null, 4.0, 3.9, 4.1, 4.2, null, 3.4, 3.3, null, null, 2.7, 3.2, null, 3.4, 3.7, null, 4.0, 3.4, null, 3.7, 2.7, null, 4.3, null, 4.3, 3.7, 3.5, 4.3, null, 3.6, 2.2, 4.2, 3.8, null, 4.3, 4.1, 3.1, 3.1, 3.9, 4.4, 3.9, 3.5, 3.4, 2.8, 3.8, null, 3.8, null, 3.0, 4.2, null, 2.9, 4.2, 4.4, null, 3.3, 3.3, 4.0, 3.9, null, 4.2, 3.2, null, 3.5, 3.1, null, null, 4.1, null, 3.8, 4.6, null, null, 4.2, 3.2, 3.5, 3.7, 3.8, 3.8, 3.4, 4.0, 3.3, 3.2, 3.7, 3.5, null, 3.8, 3.5, 3.5, 4.3, null, 3.9, 3.8, 4.1, 4.2, 3.6, 3.9, 3.9, 3.1, 3.5, 3.7, null, 4.2, 4.5, 3.7, null, 4.0, null, 3.2, 4.1, 4.1, 4.3, 3.4, 4.0, 4.1, 3.5, 3.8, 3.6, 3.9, 2.7, 4.3, 4.0, 3.8, 4.1, 3.9, 3.5, 3.6, 3.9, 3.5, 4.1, 3.6, 3.2, 3.2, 3.7, null, null, 3.4, 3.6, null, 4.3, 2.4, 3.5, 3.3, 3.7, 3.2, null, 3.0, 4.0, 4.0, 3.9, 3.5, 3.9, 3.4, 3.7, null, 4.0, 3.1, null, 3.2, 3.6, 3.6, 3.7, 4.1, 3.7, 3.6, 4.3, 2.9, 4.2, 3.8, null, 4.4, 3.5, 4.0, 3.3, 3.7, 3.8, 3.4, 4.0, 3.5, null, 2.9, 3.3, 3.7, 3.9, 3.6, 3.9, 3.3, 3.4, 3.7, 4.4, 2.8, 3.5, 3.9, 2.9, null, 4.0, null, 3.8, 2.8, 4.3, 3.6, 2.8, 3.5, 3.2, 3.8, 3.3, null, 2.9, 4.2, 2.7, null, 3.4, null, 3.7, 4.1, null, 4.4, 2.7, null, 3.6, 3.1, null, 4.3, 2.8, 3.8, 3.5, 3.2, 3.6, 3.9, 3.9, 3.6, 3.9, 4.1, null, null, 4.5, 3.3, 3.8, null, null, 3.3, 3.6, 3.5, 3.3, 4.1, null, 4.2, 3.9, 3.7, 3.9, 3.8, 3.6, 3.9, 4.4, 3.7, 3.5, 4.0, null, 3.8, 4.0, 2.9, 3.9, 3.9, 4.1, 3.7, 3.5, 3.1, null, 4.4, 3.7, 2.9, 2.6, 4.0, 2.5, 3.6, 3.3, 3.8, 3.7, 3.5, 2.8, 3.9, 3.8, 3.7, 3.0, null, 3.2, null, 4.4, 3.4, 3.7, 3.5, null, 4.3, 4.3, null, 3.4, 3.3, 2.8, 3.4, 4.2, 3.9, 3.8, 4.4, null, 4.1, 3.6, 3.4, 3.0, 3.7, 2.1, 3.7, 3.3, 3.9, null, 3.0, null, 3.0, 3.1, 2.8, 3.5, 3.4, 3.3, 4.1, null, 3.9, null, 4.1, 3.7, 3.8, 3.8, 3.0, null, 3.9, 3.3, 4.4, 3.5, 3.7, 4.3, 3.7, 4.2, 3.6, null, 4.1, null, 4.0, 3.4, 2.7, 2.9, null, 4.4, 3.6, 4.0, null, 3.1, 3.4, null, 3.3, 3.8, 3.4, 3.4, null, 4.2, 4.1, null, null, null, 3.2, 3.4, 3.9, 4.1, 2.6, 3.9, 3.0, 3.6, 4.0, 3.7, 4.3, null, null, 4.1, 3.4, 4.3, 3.6, 4.5, 3.9, 3.8, 4.1, 3.6, 3.3, 3.7, 3.8, 3.7, 3.9, 3.6, 4.2, null, 4.1, 3.9, null, 4.2, 3.8, 4.4, 3.7, null, null, 3.9, 2.9, 3.3, 3.1, 3.8, 4.4, 3.2, 3.6, 4.3, 4.1, 3.8, 2.9, 4.1, 3.1, 3.4, null, 3.9, 3.7, null, 4.0, 2.8, 3.3, 3.8, null, 4.0, 2.2, 4.0, null, 4.2, 3.4, 3.3, 3.6, 3.3, 3.6, 3.8, 2.8, 3.6, 3.8, null, null, 3.5, 4.0, null, null, 4.3, 3.3, 3.4, 3.8, 3.7, 3.6, 4.5, null, 3.0, 3.1, 3.6, 3.7, 3.7, null, 4.0, null, 4.2, null, null, null, 3.7, 3.2, 4.3, 4.5, null, 3.2, 3.5, 3.9, 3.8, 3.5, 4.1, 3.1, 4.2, null, 3.5, 4.4, 3.8, 4.0, null, 3.1, 3.8, null, 3.9, null, 3.9, 3.8, null, null, 3.6, 3.5, 4.2, 3.9, null, 1.8, 4.0, 4.0, null, 4.1, null, 3.5, 4.1, 2.6, null, 3.7, 3.5, null, 3.8, 4.5, null, 3.3, 3.2, 2.9, null, null, 4.2, 3.7, 3.9, 3.8, 4.0, 3.9, 3.1, 4.3, 4.3, 4.7, 3.8, 2.9, 4.1, null, 4.1, 4.2, null, 3.8, 4.4, 3.8, 3.7, 3.8, 4.0, null, 2.8, null, 4.4, 3.7, null, 3.4, 4.0, 4.1, null, null, 3.7, 2.9, null, 3.1, 4.4, 3.3, 3.3, 4.0, 3.8, 3.6, null, 4.4, 4.6, 4.3, 3.3, 3.8, 4.0, null, 4.1, 3.9, 3.8, 3.7, 3.3, 4.2, 4.2, 2.6, null, 3.5, 3.1, 2.8, null, 4.1, 4.0, 3.1, 3.4, 3.5, 3.4, 4.3, 3.7, null, null, null, 3.9, null, 4.3, 3.9, 4.1, 3.8, 4.3, 2.6, 3.6, 3.4, 4.2, null, 4.0, null, 3.7, 3.6, 3.6, 3.6, 4.4, 3.5, 4.4, 3.5, 4.5, null, null, 3.6, 3.9, 3.1, null, 4.4, 3.4, null, 3.9, 3.8, 3.7, null, 4.1, 3.4, 3.7, 4.0, 3.5, 4.2, 4.4, 3.5, null, 3.9, 3.2, 4.0, 3.5, 3.4, 4.4, 3.7, 4.0, null, 4.1, null, 2.8, 3.9, 3.8, 4.0, 2.8, 3.7, 3.5, null, 2.3, null, null, 4.1, 3.7, 3.7, 2.8, 3.3, 3.0, null, null, 3.6, 4.2, 3.8, 3.1, 3.0, 4.2, 3.1, 3.3, 3.7, null, null, 3.8, 4.1, 3.9, null, 4.1, null, 3.7, 3.5, 3.7, 3.7, 3.8, 3.5, 2.8, 3.9, 3.2, 3.2, 4.0, 3.1, 3.9, null, 4.2, 4.4, 4.4, 2.8, 4.2, 2.8, 3.1, 3.7, 2.8, 4.0, 3.8, 3.6, 2.9, 3.8, 4.2, 4.5, 3.6, null, null, null, 3.7, null, 3.2, 3.4, null, 3.2, 4.3, 3.8, null, 3.7, 3.4, 4.0, 4.0, 3.4, 2.5, 3.3, 3.8, 4.1, 3.4, 4.5, 3.8, 4.6, 3.1, 4.4, 3.2, 2.7, 3.7, 3.6, 3.4, null, 2.9, null, 4.0, 3.2, null, 4.0, 3.8, 4.2, 3.5, 2.8, null, 3.8, 4.6, null, 3.4, 3.0, null, null, null, null, 4.1, 4.5, 4.2, 3.5, 3.4, null, null, null, 3.0, 3.3, 3.2, 3.9, 3.6, null, 4.2, 4.0, 2.9, null, 3.4, 3.4, null, 3.8, 3.8, 3.7, 4.0, 3.6, 4.2, 3.7, 4.5, 4.1, 3.2, null, 4.0, 3.0, 4.0, 3.2, 2.9, 4.3, 4.0, 4.7, null, 4.0, 3.2, 4.0, 3.9, 4.1, 3.3, 3.7, null, null, 3.8, null, 3.1, 4.0, 3.6, null, 3.6, 3.8, 3.4, 3.3, 3.2, 4.5, null, null, 4.1, 3.6, 4.2, 3.0, null, null, 4.3, 4.1, 2.8, 3.6, 3.1, 3.8, 4.0, 4.1, null, 4.2, 4.1, 3.7, 2.6, 3.2, 3.3, 2.8, 3.9, null, 3.9, 3.1, 3.8, null, 3.1, 4.3, null, 4.3, 3.6, 3.9, null, 3.2, 3.6, null, 3.6, null, 4.1, 4.0, 4.3, 3.3, 3.3, null, null, 3.2, 3.5, 3.6, 3.5, 2.9, 3.9, 3.9, 3.4, 4.2, 3.8, 3.4, 3.6, 4.0, 4.1, 3.6, 4.2, 3.7, 2.7, 3.7, 3.3, 3.1, null, 2.8, 3.9, 4.3, null, 3.3, 3.7, null, 3.7, null, 3.0, 4.4, null, 3.4, 4.1, 3.6, 3.1, 4.0, 4.1, null, 3.6, 4.4, 2.9, 3.8, 4.0, 3.6, 3.6, 3.7, 3.8, 3.6, null, 3.5, null, 4.1, null, null, null, null, 4.3, 3.2, 3.3, 3.4, 3.9, 4.1, null, 3.0, null, null, 3.2, 3.4, 4.1, 4.0, null, 3.7, 3.3, 3.3, null, 4.0, null, 3.1, 3.2, 3.8, 3.2, 3.9, 4.3, 3.4, 3.7, 3.8, 3.2, 3.9, 3.2, null, 4.4, 3.3, 3.6, 3.3, null, 3.6, null, 3.9, null, 4.0, 3.8, 3.9, null, 3.8, null, 4.1, 4.2, 3.5, 3.8, null, null, 3.9, 3.6, 3.7, 4.5, 3.9, 3.2, 3.5, 3.3, 3.8, 3.1, 3.3, 2.3, 3.3, 3.7, null, 2.6, 3.4, 3.1, null, 3.6, 3.8, 4.0, 3.8, null, 3.2, null, null, 3.8, 3.6, 3.7, 3.6, 3.6, 4.0, 3.5, 3.8, 3.6, 3.6, 4.1, 4.1, null, 3.3, 3.1, null, 3.1, 3.6, 4.0, 3.4, 3.7, 3.9, 3.3, null, 3.6, 4.3, 3.2, 3.8, 3.7, 4.2, 3.3, 3.9, 3.8, 3.6, 3.6, 3.8, 4.1, 3.8, 4.0, null, 3.6, 3.9, null, 2.7, 4.4, 3.2, 3.8, 3.7, null, 2.6, null, 4.1, 3.7, 3.6, 3.5, null, null, 3.3, 3.0, 3.9, 3.6, 3.6, 3.2, null, 3.9, 3.5, 4.3, 3.2, null, 3.6, 3.2, 2.9, 3.6, 3.8, null, 3.2, 3.8, 4.1, null, 4.2, null, 3.3, 3.7, 3.6, null, null, 4.3, 3.6, 3.1, 3.9, 4.1, 4.1, 4.2, 3.9, 4.1, 4.0, 3.4, 4.5, null, 3.9, 3.4, 3.3, 4.1, 3.5, 4.0, null, 4.2, 2.6, null, 3.5, 3.5, 2.9, 3.8, 3.6, 3.7, 3.4, 4.6, 3.6, 3.8, 3.5, 4.3, 3.5, 3.1, 3.7, null, null, 4.3, 4.4, 3.9, 3.7, 4.1, 3.7, 3.5, 3.8, null, 3.0, null, 3.5, 3.0, 4.2, 3.2, 3.3, 3.2, 3.8, 3.5, 3.7, 4.5, 3.7, 2.9, 3.7, 3.6, 3.1, 3.2, null, null, 3.6, 3.5, 3.4, null, 4.1, 3.4, null, 3.3, 4.4, 3.4, null, 4.1, 3.4, 3.7, 4.3, 3.4, 3.3, 3.5, null, 3.5, 3.9, 4.0, 3.7, 4.4, 3.7, 3.3, 4.1, 2.8, 3.8, 3.9, 4.5, 3.3, null, 3.3, null, 3.7, 4.3, 3.9, 3.9, null, 3.1, null, 2.7, 3.7, 3.3, 4.3, 4.0, 3.2, null, 4.2, 4.3, 3.4, null, 3.8, null, null, 3.6, 4.3, 3.2, null, 3.5, 3.6, null, 2.8, 3.1, 2.9, 3.7, 3.6, null, 3.9, 3.5, 3.0, 3.8, 4.3, 3.4, 4.0, 3.9, 3.9, null, 3.6, 3.6, 4.2, null, null, 3.8, 3.0, 2.9, 4.3, 3.6, 3.7, 3.9, 4.5, 3.4, null, 4.2, 2.8, 4.2, null, 4.1, 3.0, 3.0, 4.6, 4.3, 3.3, 4.2, 3.6, null, 3.7, 3.0, 4.4, 4.1, 3.6, 4.2, 4.1, 3.3, 4.2, 4.1, null, 3.7, 3.8, 4.1, 3.9, 4.2, 4.1, 4.0, null, 3.6, 4.0, 3.5, 4.2, 3.9, 3.3, 4.0, 3.7, null, 3.3, 3.6, 3.9, 3.5, 3.6, 3.4, 3.4, 3.5, 3.3, 4.0, 3.7, 3.6, 4.3, 3.3, 2.8, 3.7, 3.8, 2.9, 4.2, 3.2, 4.0, null, 3.8, 3.9, null, 4.5, 3.9, 3.8, 3.3, 4.2, 4.2, 3.4, 3.9, 3.1, 3.8, 3.5, null, 3.0, 3.3, null, 3.7, 4.0, 3.6, 3.0, 3.6, null, 3.5, 4.0, null, 3.3, 3.7, 2.8, 3.4, 3.1, null, 4.0, 4.4, 3.8, 3.9, null, 2.8, 3.4, 3.8, 3.7, 4.2, 4.0, 3.8, 4.6, 3.1, 3.7, 4.0, 2.6, 3.5, 4.3, 3.9, 3.5, 3.5, 3.9, 3.5, null, 3.1, 4.2, 4.1, 3.8, 3.2, 3.3, 3.9, 3.4, 4.2, 3.7, 4.3, 2.9, 3.8, 3.8, 3.7, 4.1, 4.0, 4.3, 2.5, 4.3, 3.8, 3.1, 3.9, null, null, 3.5, 3.7, 2.9, 4.2, 4.4, 3.9, null, 3.9, 4.1, 3.4, null, 3.6, 3.9, 3.9, null, 3.2, 4.1, 3.2, 4.5, 3.5, 3.7, 3.1, 3.0, 4.3, null, 3.9, 4.3, 3.7, 3.9, 4.0, null, 3.9, 2.8, 4.0, 3.5, 4.0, 3.5, 3.4, 3.1, null, 3.1, 3.9, 4.3, 3.9, 3.8, 3.4, 4.2, null, 4.1, 4.2, 3.8, 3.1, 4.2, 3.7, 3.6, 3.0, 3.5, 3.4, 3.6, 3.6, 3.0, 4.2, 3.8, 2.2, 4.2, 3.5, 3.2, 4.1, 4.1, 2.9, 4.1, 3.8, 4.2, 3.8, 4.0, 3.8, 4.3, 3.8, null, 3.8, null, 3.7, null, null, 3.3, null, 3.8, 3.6, 2.9, 3.8, 3.4, 4.1, null, 4.0, 4.4, 3.9, 4.1, 4.4, 3.1, 3.6, null, 3.9, 3.1, 4.3, 3.5, 4.4, 2.9, 3.1, 4.1, 3.3, null, 3.8, null, 3.9, null, 3.8, 4.1, 4.1, 3.8, 4.0, 4.3, null, 3.2]}],
                        {"barmode": "overlay", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Restaurant's Rating Distribution", "x": 0.5}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true, "title": {"text": "Rating"}}, "yaxis": {"title": {"text": "Count"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('d7eb2050-02fc-4560-8c22-fbe2066367b5');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** `Rating` variable is following normal distribution (almost)


```python
plotly_histogram_chart(restaurant_df["no_of_votes"], x_title="# Votes", color="Red")
```


<div>


            <div id="ab50ee70-318b-4238-9515-70a6c22d9eca" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("ab50ee70-318b-4238-9515-70a6c22d9eca")) {
                    Plotly.newPlot(
                        'ab50ee70-318b-4238-9515-70a6c22d9eca',
                        [{"marker": {"color": "Red"}, "opacity": 0.75, "type": "histogram", "x": [0, 21, 131, 3236, 225, 402, 9, 712, 64, 46, 184, 0, 7, 13, 291, 89, 289, 0, 1214, 207, 433, 0, 280, 0, 121, 203, 0, 44, 0, 9, 214, 3468, 0, 114, 163, 94, 68, 263, 166, 208, 11, 7, 14, 1084, 0, 1258, 4, 99, 9, 0, 94, 960, 12, 19, 2487, 0, 100, 1026, 0, 47, 0, 2049, 1439, 10, 34, 237, 5, 604, 0, 18, 86, 833, 57, 0, 15, 370, 72, 75, 185, 89, 11, 96, 152, 894, 12, 171, 0, 0, 42, 23, 155, 22, 787, 74, 13, 0, 498, 0, 0, 14, 53, 0, 10, 49, 0, 782, 434, 0, 167, 29, 210, 21, 0, 51, 51, 18, 1426, 3870, 189, 11, 43, 103, 452, 165, 18, 29, 69, 0, 0, 0, 76, 46, 6, 241, 178, 181, 888, 82, 241, 0, 131, 0, 41, 40, 33, 9, 106, 11, 88, 2634, 719, 861, 89, 0, 0, 6, 285, 67, 3230, 0, 57, 0, 0, 0, 117, 135, 17, 503, 242, 18, 0, 484, 82, 49, 0, 0, 15, 0, 14, 8, 136, 0, 480, 21, 337, 0, 616, 218, 269, 0, 110, 118, 375, 23, 31, 142, 79, 21, 334, 476, 5, 1154, 0, 1225, 39, 130, 27, 0, 101, 32, 132, 7330, 0, 0, 131, 9, 1808, 53, 0, 0, 28, 0, 410, 0, 0, 28, 32, 324, 347, 201, 20, 34, 327, 41, 592, 791, 154, 822, 195, 0, 942, 1847, 0, 23, 163, 289, 12, 16, 179, 235, 59, 0, 60, 0, 597, 240, 0, 47, 59, 0, 8, 21, 9, 12, 994, 994, 421, 12, 0, 30, 11, 0, 238, 36, 47, 759, 27, 17, 16, 0, 14, 57, 21, 168, 420, 217, 253, 1708, 120, 97, 0, 0, 33, 12, 53, 1048, 108, 18, 450, 0, 0, 19, 0, 47, 11, 104, 38, 11, 355, 232, 632, 23, 48, 77, 189, 638, 38, 111, 41, 8, 183, 5, 8, 5, 85, 523, 27, 291, 402, 0, 14, 93, 233, 97, 800, 39, 2291, 0, 7, 27, 9, 41, 59, 138, 548, 23, 151, 0, 44, 24, 0, 73, 10, 10, 48, 11, 552, 38, 27, 131, 971, 0, 289, 494, 0, 176, 74, 181, 49, 229, 17, 519, 76, 1053, 152, 0, 11, 185, 1241, 1269, 0, 5, 1543, 889, 0, 0, 17, 1289, 13, 559, 49, 66, 0, 0, 100, 178, 272, 10, 0, 0, 427, 8, 504, 15, 5, 0, 487, 0, 42, 67, 89, 148, 324, 34, 170, 11, 0, 9, 0, 9, 0, 66, 138, 38, 4, 253, 11, 4, 0, 0, 8, 212, 0, 58, 110, 24, 71, 1229, 294, 136, 790, 129, 57, 185, 21, 27, 4, 35, 60, 50, 7, 16, 0, 553, 570, 89, 100, 196, 0, 132, 1461, 7, 261, 13, 12, 197, 0, 0, 0, 10, 759, 18, 270, 13, 278, 282, 28, 290, 86, 385, 0, 57, 4, 507, 399, 0, 118, 56, 0, 531, 108, 0, 32, 621, 20, 0, 13, 109, 9, 17, 3621, 0, 450, 25, 1077, 776, 0, 64, 20, 837, 110, 6, 403, 91, 155, 1792, 128, 0, 66, 7854, 23, 1087, 0, 24, 0, 499, 50, 448, 226, 0, 9, 5, 0, 0, 53, 70, 0, 10, 14, 0, 57, 6, 0, 88, 73, 0, 3592, 0, 429, 41, 41, 345, 0, 91, 479, 1703, 38, 0, 972, 559, 85, 10, 35, 571, 728, 33, 46, 63, 33, 0, 37, 0, 11, 233, 0, 67, 444, 2055, 0, 4, 7, 605, 55, 0, 476, 4, 0, 6, 28, 0, 0, 100, 0, 12, 1095, 0, 0, 3236, 5, 8, 48, 48, 94, 4, 243, 59, 17, 42, 436, 0, 80, 19, 43, 165, 0, 164, 2332, 125, 676, 26, 448, 247, 11, 14, 92, 0, 1345, 727, 115, 0, 362, 0, 4, 701, 364, 446, 48, 287, 1750, 435, 83, 101, 46, 182, 290, 508, 64, 28, 48, 433, 16, 45, 31, 110, 57, 6, 4, 47, 0, 0, 6, 199, 0, 512, 392, 7, 9, 33, 7, 0, 7, 147, 783, 53, 24, 885, 19, 19, 0, 53, 68, 0, 20, 104, 59, 31, 289, 24, 199, 189, 539, 58, 34, 0, 290, 45, 553, 6, 96, 187, 61, 2164, 11, 0, 27, 14, 69, 56, 13, 914, 110, 4, 19, 192, 148, 16, 1214, 4, 0, 113, 0, 182, 26, 225, 23, 92, 29, 4, 156, 10, 0, 28, 53, 53, 0, 276, 0, 50, 71, 0, 180, 48, 0, 10, 151, 0, 3468, 25, 32, 229, 6, 25, 281, 468, 29, 18, 3238, 0, 0, 3163, 9, 22, 0, 0, 4, 8, 185, 17, 18, 0, 800, 130, 21, 1049, 120, 19, 145, 109, 20, 283, 1310, 0, 56, 220, 539, 1142, 747, 152, 106, 52, 8, 0, 635, 36, 18, 254, 61, 157, 61, 9, 152, 168, 12, 34, 420, 37, 514, 8, 0, 4, 0, 136, 58, 15, 13, 0, 294, 251, 0, 17, 6, 137, 5, 984, 718, 122, 725, 0, 51, 19, 13, 88, 242, 479, 51, 4, 463, 0, 39, 0, 13, 106, 420, 140, 10, 4, 197, 0, 326, 0, 786, 68, 22, 146, 19, 0, 462, 7, 203, 53, 60, 171, 96, 142, 42, 0, 1320, 0, 76, 153, 442, 6, 0, 2662, 10, 200, 0, 4, 8, 0, 7, 34, 11, 76, 0, 337, 42, 0, 0, 0, 5, 78, 510, 399, 283, 118, 36, 62, 48, 21, 549, 0, 0, 266, 5, 168, 15, 3486, 18, 31, 520, 16, 7, 19, 71, 19, 84, 12, 2714, 0, 1156, 125, 0, 1177, 66, 3712, 37, 0, 0, 96, 33, 16, 113, 92, 1972, 27, 42, 1187, 1359, 69, 57, 159, 10, 32, 0, 55, 24, 0, 55, 56, 44, 61, 0, 783, 406, 126, 0, 1745, 24, 4, 21, 34, 94, 591, 70, 11, 46, 0, 0, 13, 253, 0, 0, 620, 5, 7, 29, 32, 10, 236, 0, 16, 5, 195, 19, 17, 0, 332, 0, 984, 0, 0, 2508, 47, 4, 1370, 819, 0, 4, 10, 72, 25, 14, 38, 7, 1175, 0, 13, 4884, 500, 25, 0, 23, 362, 0, 331, 0, 536, 251, 0, 0, 31, 70, 1647, 250, 0, 225, 87, 65, 0, 558, 0, 11, 29, 100, 0, 47, 178, 0, 22, 2073, 0, 9, 5, 25, 0, 0, 70, 102, 951, 67, 568, 485, 9, 241, 456, 277, 464, 89, 2450, 0, 45, 166, 0, 273, 1106, 623, 220, 229, 155, 0, 38, 0, 2389, 19, 0, 11, 1871, 500, 4, 0, 257, 48, 0, 30, 949, 7, 6, 113, 111, 16, 0, 100, 4694, 476, 7, 21, 386, 0, 1858, 109, 36, 50, 4, 34, 2720, 60, 0, 177, 17, 110, 0, 383, 372, 11, 17, 9, 4, 3624, 130, 0, 0, 0, 184, 0, 1187, 357, 511, 12, 450, 158, 12, 9, 133, 0, 76, 0, 111, 26, 79, 35, 1856, 15, 790, 15, 1878, 0, 0, 34, 680, 7, 0, 191, 12, 0, 39, 46, 63, 0, 111, 83, 44, 130, 221, 3236, 1804, 21, 0, 79, 13, 147, 11, 6, 488, 30, 410, 0, 970, 0, 19, 236, 18, 2852, 137, 434, 10, 0, 176, 0, 0, 1858, 73, 34, 34, 4, 289, 0, 0, 34, 819, 109, 14, 18, 505, 9, 12, 17, 0, 0, 19, 115, 123, 0, 360, 0, 468, 180, 34, 47, 311, 13, 14, 924, 16, 7, 863, 19, 191, 0, 1223, 118, 2867, 261, 1431, 211, 7, 323, 302, 25, 37, 14, 54, 194, 860, 3843, 159, 0, 0, 0, 28, 0, 7, 6, 0, 64, 995, 63, 0, 31, 8, 776, 417, 16, 47, 6, 149, 59, 15, 2198, 16, 866, 122, 2041, 4, 82, 126, 192, 5, 0, 8, 0, 419, 5, 0, 789, 14, 2032, 10, 38, 0, 39, 2332, 0, 7, 98, 0, 0, 0, 0, 167, 1508, 432, 185, 9, 0, 0, 0, 16, 4, 4, 185, 180, 0, 175, 76, 79, 0, 16, 215, 0, 208, 69, 67, 790, 211, 38, 39, 3991, 81, 4, 0, 39, 197, 74, 22, 166, 2577, 233, 4811, 0, 213, 835, 132, 267, 853, 4, 195, 0, 0, 109, 0, 6, 1203, 42, 0, 24, 94, 6, 11, 6, 1238, 0, 0, 1785, 23, 275, 20, 0, 0, 108, 337, 92, 24, 117, 70, 169, 129, 0, 376, 61, 33, 72, 17, 4, 100, 681, 0, 82, 166, 22, 0, 8, 2304, 0, 1196, 70, 247, 0, 21, 140, 0, 11, 0, 41, 151, 2447, 4, 12, 0, 0, 20, 7, 38, 21, 142, 54, 783, 31, 239, 44, 9, 154, 610, 109, 9, 1920, 15, 57, 50, 7, 245, 0, 19, 215, 84, 0, 4, 22, 0, 14, 0, 11, 679, 0, 9, 1324, 198, 330, 25, 894, 0, 45, 864, 57, 38, 127, 25, 539, 30, 257, 11, 0, 31, 0, 845, 0, 0, 0, 0, 1332, 23, 4, 17, 150, 161, 0, 15, 0, 0, 11, 7, 271, 153, 0, 25, 47, 111, 0, 160, 0, 12, 68, 69, 4, 191, 570, 99, 26, 158, 4, 88, 4, 0, 1865, 4, 23, 82, 0, 42, 0, 370, 0, 76, 194, 599, 0, 232, 0, 1850, 470, 9, 27, 0, 0, 42, 112, 25, 3987, 125, 10, 50, 5, 71, 5, 12, 235, 5, 47, 0, 84, 6, 25, 0, 156, 258, 427, 148, 0, 5, 0, 0, 214, 136, 126, 12, 55, 958, 15, 94, 15, 23, 205, 201, 0, 4, 28, 0, 11, 13, 154, 5, 24, 49, 218, 0, 17, 361, 7, 37, 66, 164, 395, 60, 41, 15, 23, 31, 1503, 187, 101, 0, 24, 240, 0, 29, 63, 15, 75, 82, 0, 30, 0, 145, 21, 221, 13, 0, 0, 4, 11, 125, 17, 68, 6, 0, 122, 29, 1052, 4, 0, 28, 17, 17, 5, 63, 0, 16, 32, 520, 0, 957, 0, 11, 25, 61, 0, 0, 1721, 92, 44, 65, 237, 563, 515, 358, 600, 124, 5, 188, 0, 228, 946, 5, 785, 21, 284, 0, 289, 74, 0, 25, 15, 19, 90, 39, 382, 7, 4694, 52, 31, 21, 429, 16, 22, 8, 0, 0, 754, 69, 190, 25, 2773, 14, 11, 30, 0, 21, 0, 15, 34, 176, 4, 23, 9, 46, 43, 145, 1013, 16, 45, 259, 9, 6, 4, 0, 0, 70, 54, 9, 0, 2461, 54, 0, 4, 4460, 4, 0, 292, 15, 37, 454, 5, 39, 89, 0, 6, 652, 42, 25, 2182, 127, 5, 399, 161, 110, 224, 194, 42, 0, 12, 0, 41, 1135, 823, 135, 0, 7, 0, 31, 11, 9, 2230, 578, 6, 0, 226, 62, 257, 0, 117, 0, 0, 16, 286, 5, 0, 19, 14, 0, 511, 150, 21, 63, 68, 0, 43, 12, 36, 377, 654, 12, 339, 66, 59, 0, 23, 11, 570, 0, 0, 94, 31, 243, 3126, 32, 195, 218, 3471, 8, 0, 1428, 270, 115, 0, 2317, 110, 39, 206, 687, 13, 1476, 97, 0, 28, 25, 2861, 221, 112, 2034, 295, 16, 980, 847, 0, 16, 191, 573, 501, 744, 724, 409, 131, 19, 932, 20, 215, 134, 47, 1310, 49, 0, 4, 123, 42, 56, 16, 39, 225, 13, 4, 118, 329, 22, 1143, 7, 102, 35, 151, 77, 624, 5, 28, 0, 23, 331, 0, 804, 33, 85, 6, 2049, 713, 20, 1190, 43, 50, 12, 0, 10, 24, 0, 181, 632, 383, 404, 14, 0, 14, 130, 0, 15, 270, 36, 11, 5, 0, 749, 4079, 23, 131, 0, 185, 17, 38, 17, 240, 314, 111, 2490, 6, 15, 152, 408, 8, 2139, 169, 11, 11, 118, 28, 0, 121, 386, 568, 27, 4, 8, 347, 8, 4733, 30, 7544, 24, 200, 56, 31, 362, 76, 1048, 23, 379, 41, 10, 38, 0, 0, 14, 27, 26, 176, 7210, 888, 0, 66, 164, 7, 0, 39, 1791, 2322, 0, 17, 313, 21, 3991, 33, 26, 38, 6, 548, 0, 727, 1559, 30, 54, 856, 0, 237, 58, 40, 10, 60, 75, 48, 104, 0, 105, 592, 1593, 32, 63, 23, 583, 0, 60, 876, 376, 23, 174, 51, 24, 16, 140, 8, 8, 108, 9, 53, 91, 409, 1901, 6, 6, 159, 493, 93, 184, 149, 858, 69, 34, 706, 324, 377, 0, 122, 0, 21, 0, 0, 11, 0, 101, 68, 18, 410, 422, 176, 0, 89, 568, 61, 262, 751, 253, 112, 0, 851, 13, 61, 4, 1227, 20, 68, 988, 11, 0, 273, 0, 96, 0, 693, 341, 95, 214, 1013, 2039, 0, 4]}],
                        {"barmode": "overlay", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Restaurant's # Votes Distribution", "x": 0.5}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true, "title": {"text": "# Votes"}}, "yaxis": {"title": {"text": "Count"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('ab50ee70-318b-4238-9515-70a6c22d9eca');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** `# Votes` variable is following right skewed distribution


```python
plotly_histogram_chart(restaurant_df["cost_for_two_people"], x_title="Cost for Two People", color="Red")
```


<div>


            <div id="65ce6215-7323-4b8c-b713-a39a4e6ce68d" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("65ce6215-7323-4b8c-b713-a39a4e6ce68d")) {
                    Plotly.newPlot(
                        '65ce6215-7323-4b8c-b713-a39a4e6ce68d',
                        [{"marker": {"color": "Red"}, "opacity": 0.75, "type": "histogram", "x": ["200", "200", "230", null, "800", "1,200", "250", "1,000", "250", "450", "350", "300", "200", "250", "500", "350", "800", "300", "1,200", "1,000", "800", "400", "1,500", "150", "600", "900", "400", "100", "400", "300", "600", "1,300", "300", "1,500", "500", "300", "450", "700", "400", "400", "200", "400", "500", "600", "200", "1,000", "600", "200", "400", "300", "500", "600", "200", "700", "900", "150", "300", "800", "300", "200", "400", "1,600", "2,200", "100", "200", null, "1,000", "1,800", "300", "400", "500", "700", "200", "250", "100", "1,000", "300", "500", "600", "400", "300", "300", "700", "750", "700", "400", "400", "300", "800", "300", "400", "400", "650", "600", "400", "400", "2,100", "300", "350", "250", "300", "100", "250", "300", "150", "1,500", "800", "800", "1,700", "400", "800", "200", "150", "300", "500", "600", "1,700", "1,600", "400", "50", "600", "700", "600", "3,000", "500", "300", "700", "150", "600", "300", "400", "350", "200", "500", "800", "250", "650", "200", "500", "550", "700", "300", "700", "300", "600", "300", "600", "200", "500", "500", "950", "1,800", "200", "300", "100", "300", "800", "600", "1,600", "450", "400", "150", "300", "350", "2,000", "800", "800", "700", "400", "200", "700", "3,000", "200", "200", "300", "1,500", "400", "200", "200", "150", "600", "400", "450", "200", "1,000", "200", "500", "400", "300", "300", "200", "1,000", "3,000", "500", "500", "550", "500", "700", "550", "1,400", "400", "1,100", "600", "400", "950", "300", "600", "900", "250", "700", "300", "1,400", "150", "300", "400", "550", "1,000", "750", "600", "1,500", "200", "300", "500", "1,200", "200", "500", "200", "700", "1,000", "400", "400", "200", "500", "100", "1,300", "1,000", "200", "1,000", "3,000", "200", "1,300", "600", "250", "500", "500", "400", "200", "250", "300", "1,000", "300", "200", "400", "200", "2,500", "600", "150", "600", "200", "150", "600", "150", "300", "150", "1,300", "1,300", "800", "200", "200", "800", "250", "300", "800", "150", "200", "1,200", "550", "400", "1,100", "300", "200", "350", "250", "2,500", "700", "1,500", "1,000", "1,300", "300", "2,400", "500", "300", "500", "150", "350", "1,600", "450", "100", "400", "250", "100", "400", "200", "100", "200", "1,200", "900", "200", "500", "1,000", "500", "350", "250", "250", "300", "400", "350", "800", "200", "250", "550", "200", "600", "600", "600", "2,500", "300", "750", "1,000", "500", "300", "200", "500", "650", "750", "700", "1,400", "300", "300", "1,000", "700", "400", "300", "500", "1,200", "800", "1,350", "400", "500", "550", "250", "250", "100", "500", "1,500", "1,000", "800", "400", "200", "800", "1,500", "500", "400", "800", "100", "400", "400", "550", "400", "400", "300", "400", "500", "500", "300", "500", "300", "300", "1,500", "1,000", "400", "100", "1,200", "800", "250", "200", "400", "1,100", "200", "1,100", "300", "800", "250", "650", "1,900", "700", "600", "1,000", "250", "500", "500", "200", "350", "350", "450", "200", "700", "200", "400", "600", "1,400", "600", "180", "150", "1,300", "200", "350", "300", "300", "250", "300", "800", "800", "900", "200", "1,000", "250", "300", "400", "500", "300", "400", "200", "300", "200", "500", "1,000", "700", "600", "1,500", "1,000", "150", "250", "500", "300", "300", "200", "100", "400", "400", "100", "200", "400", "900", "1,200", "300", "150", "600", "400", "750", "1,500", "200", "500", "500", "500", "1,000", "300", null, "150", "250", "500", "500", "4,000", "350", "200", "500", "350", "450", "600", "1,500", "200", "400", "250", "1,000", "1,000", "500", "500", "500", "500", "800", "700", "300", "500", "3,000", "550", "500", "600", "300", "150", "200", "400", "300", "700", "450", "400", "1,000", "200", "400", "500", "1,100", "500", "500", "400", "550", "300", "1,200", "600", "250", "150", "1,800", "300", "550", "100", "550", "400", "850", "500", "700", "200", "200", "300", "400", "500", "300", "600", "250", "400", "500", "300", "300", "500", "400", "300", "500", "600", "500", "2,100", "500", "3,400", "700", "600", "400", "300", "600", "650", "650", "400", "1,000", "750", "500", "400", "500", "350", "4,000", "800", "400", "200", "600", "300", "1,500", "400", "150", "700", "300", "100", "500", "100", "1,200", "300", "200", "400", "800", "300", "500", "1,700", "500", "200", "400", "250", "200", "350", "2,400", "400", "300", "1,500", "500", "500", null, "350", "250", "350", "150", "250", "350", "500", "300", "400", "2,000", "500", "200", "800", "600", "150", "2,800", "300", "400", "700", "900", "650", "300", "600", "650", "300", "150", "400", "200", "1,700", "1,500", "550", "550", "1,000", "400", "150", "600", "550", "2,000", "300", "150", "1,000", "500", "400", "200", "300", "450", "500", "1,000", "250", "2,000", "3,000", "500", "500", "550", "200", "150", "1,000", "200", "200", "500", "400", "150", "600", "500", "200", "400", "500", "200", "300", "400", "250", "300", "300", "400", "750", "600", "300", "900", "700", "300", "400", "400", "550", "350", "700", "700", "300", "400", "850", "500", "500", "400", "500", "250", "350", "250", "400", "400", "900", "300", "1,100", "500", "250", "1,000", "500", "400", "400", "600", "650", "200", "700", "1,000", "300", "400", "400", "1,000", "750", "150", "1,200", "100", "300", "400", "150", "600", "400", "2,000", "350", "600", "2,200", "450", "400", "250", "600", "200", "400", "600", "500", "250", "300", "550", "400", "400", "250", "650", "300", "300", "450", "400", "600", "500", "350", "350", "450", "150", "500", "700", "100", "200", "800", "200", "80", "1,500", "300", "500", "200", "700", "400", "800", "800", "500", "200", "200", "750", "400", "250", "600", "600", "400", "600", "120", "200", "350", "750", "150", "500", "500", "500", "500", "1,500", "600", "600", "300", "250", "300", "400", "300", "400", "1,050", "300", "450", "500", "250", "1,500", "600", "250", "600", "800", "400", "600", "300", "150", "250", "200", "1,200", "900", "400", "400", "300", "850", "200", "400", "200", "300", "300", "400", "1,000", "600", "700", "1,700", "250", "200", "400", "300", "500", "400", "650", "300", "400", "800", "450", "400", "400", "400", "450", "700", "600", "500", "300", "300", "250", "500", "800", "1,500", "1,000", "150", "600", "500", "300", "600", "1,000", "800", "350", "400", "500", "1,100", "500", "400", "150", "1,200", "550", "300", "800", "900", "250", "300", "600", "100", "300", null, "350", "200", "400", "200", "350", "800", "450", "200", "200", "150", "300", "250", "200", "300", "450", "700", "700", "600", "800", "200", "400", "1,500", "700", "1,000", "400", "1,000", "1,000", "400", "500", "350", "1,300", "300", "300", "300", "300", "350", "1,100", "500", "200", "300", "500", "1,000", "400", "500", "600", "150", "1,300", "200", "1,300", "400", "250", "300", "1,300", "100", "150", "1,000", "700", "800", "450", "400", "550", "700", "700", "400", "300", "150", "350", "300", "400", "500", "300", "600", "1,200", "200", "1,000", "450", "750", "1,000", "300", "300", "1,400", "600", "200", "300", "500", "300", "700", "650", "200", "2,000", "400", "400", "300", "300", "400", "100", "3,400", "200", "200", "800", "200", "300", "1,250", "400", "150", "400", "500", "200", "300", "400", "300", "300", "1,000", "800", "250", "900", "250", "400", "1,000", "800", "1,500", "300", "1,300", "2,500", "350", "400", "500", "150", "1,300", "450", "300", "750", null, "750", "800", "450", "600", "300", "200", "200", "450", "350", "300", "500", "800", "600", "600", "500", "300", "1,200", "200", "350", "300", "150", "200", "300", "150", "500", "300", "150", "300", "400", "650", "1,200", "500", "200", "150", "300", "200", "200", "400", "300", "1,100", "500", "1,000", "700", "250", "100", "1,500", "2,000", "300", "350", "700", "400", "300", "700", "200", "900", "600", "3,000", "400", "500", "500", "600", "500", "1,000", "1,000", "300", "150", "300", "600", "600", "100", "150", "450", "400", "500", "500", "900", "200", "300", "200", "800", "100", "150", "700", "1,500", "600", "450", "850", "700", "250", "1,200", "1,500", "300", "550", "300", "300", "1,200", "900", "400", "500", "550", "500", "200", "700", "500", "400", "100", "300", "450", "400", "400", "200", "500", "300", "300", "250", "550", "500", "500", "300", "400", "300", "600", "600", "1,500", "300", "300", "150", "700", "300", "300", "200", "1,100", "400", "400", "400", "900", "300", "250", "400", "1,200", "500", "200", "1,000", "300", "200", "200", "300", "300", null, "1,400", "800", "500", "300", "400", null, "1,000", "600", "250", "1,200", "700", "400", "250", "300", "1,900", "500", "300", "300", "1,500", "200", "350", "750", "300", "1,100", "300", "400", "250", "200", "450", "400", "600", "1,200", "250", "500", "500", "200", "500", "400", "200", "300", "300", "500", "550", "450", "500", "300", "700", "300", "400", "300", "200", "700", "1,100", "300", "400", "200", "200", "800", "300", "350", "800", "150", "300", "300", "200", "250", "800", "400", "300", "400", "1,400", "1,500", "750", "700", "800", "300", "500", "850", "500", "200", "200", "300", "250", "800", "1,800", "800", "350", "600", "150", "500", "500", "500", "300", "400", "400", "250", "350", "600", "300", "150", "450", "800", "1,000", "500", "800", "350", "800", "300", "150", "1,800", "400", "1,400", "1,200", "1,200", "200", "400", "750", "450", "150", "300", "500", "800", "1,400", "100", "400", "400", "350", "1,000", "400", "500", "150", "600", "2,800", "200", "500", "800", "450", "800", "150", "500", "400", "1,900", "500", "800", "200", "500", "150", "300", "150", "250", "300", "400", "1,300", "400", "800", "300", "400", "200", "600", "1,000", "500", "650", "700", "150", "1,300", "400", "200", "550", "1,600", "150", "600", "200", "500", "700", "600", "400", "250", "1,200", "500", "1,300", "800", null, "450", "250", "200", "1,400", "1,200", "400", "300", "200", "450", "500", "150", "2,000", "300", "200", "850", "1,100", "300", "150", "150", "1,700", "350", "400", "1,200", "150", "4,000", "500", "1,000", "150", "1,000", "300", "350", "250", "200", "750", "1,100", "800", "200", "350", "600", "400", "300", "400", "200", "800", "500", "400", "300", "600", "500", "200", "250", "800", "400", "1,500", "800", "450", "150", "600", "300", "250", "200", "350", "150", "500", "800", "300", "300", "250", "300", "300", "1,000", "300", "200", "400", "400", "300", "250", "600", "600", "150", "400", "1,000", "1,800", "1,000", "1,500", "300", "800", "550", "150", "300", "400", "300", "600", "200", "400", "600", "250", "600", "250", "200", "400", "1,500", "400", "400", "800", "500", "500", "1,100", "800", "200", "700", "3,000", "600", "300", "900", "1,600", "800", "400", "600", "150", "150", "300", "300", "700", "100", "500", "250", "300", "1,000", "400", "300", "400", "500", "300", "400", "600", "300", "700", "200", "250", "500", "700", "200", "200", "300", "300", "300", "700", "200", "800", "400", "700", "400", "300", "700", "400", "200", "1,300", "200", "1,000", "300", "250", "1,800", "400", "350", "500", "150", "200", "400", "200", "200", "300", "800", "1,300", "400", "500", "400", "600", "400", "200", "1,200", "200", "300", "400", "500", "400", "1,600", "500", "500", "600", "300", "300", "400", "300", "800", "400", "250", "850", "400", "300", "150", "150", "300", "300", "2,400", "150", "500", "250", "300", "1,000", "700", "600", "600", "500", "200", "500", "400", "250", "700", "450", "250", "300", null, "100", "200", "250", "300", "200", "500", "150", "400", "600", "550", "500", "200", "3,500", "200", "200", "800", "750", "400", "300", "400", "300", "300", "600", "1,500", "500", "150", "300", "300", "400", "300", "600", "300", "250", "1,000", "300", "300", "500", "150", "200", "150", "600", "700", "400", "150", "1,400", "400", "200", "800", "300", "100", "150", "250", "2,200", "1,600", "600", "400", "600", "550", "150", "300", "350", "250", "350", "350", "300", "800", "800", "300", "600", "500", "250", "250", "400", "1,300", "350", "150", "100", "1,000", "350", "800", "400", "400", "200", "350", "1,200", "200", "600", null, "400", "1,500", "500", "1,500", "150", "400", "700", "200", "150", "400", "450", "650", "600", "600", "300", "1,500", "600", "150", "450", "3,400", "600", "400", "100", "1,100", "600", "500", "300", "600", "400", "1,500", "200", "400", "600", "500", "600", "400", "300", "500", "800", "300", "400", "1,800", "450", "450", "1,000", "1,100", "600", "600", "750", "300", "200", "350", "350", "400", "600", "1,800", "800", "400", "1,000", "250", "400", "350", "150", "600", "100", "600", "150", "600", "3,000", "300", "250", "500", "400", "400", "500", "300", "300", "1,400", "700", "300", "300", "1,200", "1,400", "250", "200", "500", "250", "250", "200", "250", "800", "550", "800", "600", "300", "200", "500", "1,200", "250", "1,700", "600", "400", "500", "1,000", "250", "250", "200", "350", "80", "300", "200", "150", "550", "350", "1,000", "200", "300", "350", "750", "600", "300", "550", "200", "200", "300", "200", "500", "750", "300", "550", "200", "500", "100", "500", "700", "1,200", "350", "300", "250", "150", "300", "1,500", "400", "400", "600", "1,300", "550", "300", "1,300", "500", "400", "500", "950", "650", "400", "200", "3,000", "600", "1,500", "300", "150", "500", "300", "1,500", "1,000", "500", "1,200", "700", "400", "1,000", "1,100", "300", "250", "300", "600", "300", "750", "300", "350", "650", "350", "1,000", "550", "1,500", "550", "300", "750", "300", "250", "500", "600", "500", "1,800", "150", "500", "300", "400", "200", "250", "500", "350", "1,100", "350", "350", "350", "400", "400", "300", "250", "500", "300", null, "400", "500", "800", "300", "300", "300", "1,000", "500", "1,200", "500", "500", "250", "650", "500", "300", "300", "200", "500", "800", "600", "500", "200", "200", "250", "230", "400", "900", "500", "550", "300", "400", "250", "1,100", "1,200", null, "450", "500", "500", "400", "400", "200", "1,500", "800", "600", "1,600", "300", "200", "350", "1,000", "600", "600", "600", "250", "300", "800", "350", "250", "500", "500", "2,500", "150", "500", "500", "600", "400", "1,500", "350", "1,500", "250", "1,600", "400", "350", "400", "300", "1,600", "600", "700", "500", "900", "400", "200", "250", "300", "700", "300", null, "600", "750", "200", "200", "600", "300", "350", "400", "1,300", "700", "250", "500", "650", "600", "1,600", "600", "700", "300", "200", "1,000", "300", "800", "1,200", "900", "1,500", "650", "250", null, "600", "300", "800", "400", "500", "550", "250", "250", "400", "1,300", "550", "800", "300", "400", "700", "350", "600", "600", "600", "400", "600", "300", "250", "150", "600", "150", "400", "450", "450", "400", "1,800", "1,000", "800", "400", "250", "300", "600", "400", "550", "250", "1,700", "300", "200", "750", "180", "500", "400", null, "200", "500", "350", "150", "300", "300", "350", "300", "450", "350", "400", "800", "300", "2,500", "400", "200", "700", "3,000", "300", "300", "300", "700", "400", "700", "200", "800", "600", "550", "600", "400", "200", "900", "500", "350", "400", "750", "600", "600", "700", "1,200", "1,700", "300", "150"]}],
                        {"barmode": "overlay", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Restaurant's Cost for Two People Distribution", "x": 0.5}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true, "title": {"text": "Cost for Two People"}}, "yaxis": {"title": {"text": "Count"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('65ce6215-7323-4b8c-b713-a39a4e6ce68d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** `Cost for Two People` variable is also following right skewed distribution


```python
plotly_bubble_chart(restaurant_df["rating"], restaurant_df["no_of_votes"], restaurant_df["cost_for_two_people"])
```


<div>


            <div id="7bf1f7b2-f05a-47b1-bd3f-8d53225820ec" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("7bf1f7b2-f05a-47b1-bd3f-8d53225820ec")) {
                    Plotly.newPlot(
                        '7bf1f7b2-f05a-47b1-bd3f-8d53225820ec',
                        [{"hovertext": ["200", "200", "230", null, "800", "1,200", "250", "1,000", "250", "450", "350", "300", "200", "250", "500", "350", "800", "300", "1,200", "1,000", "800", "400", "1,500", "150", "600", "900", "400", "100", "400", "300", "600", "1,300", "300", "1,500", "500", "300", "450", "700", "400", "400", "200", "400", "500", "600", "200", "1,000", "600", "200", "400", "300", "500", "600", "200", "700", "900", "150", "300", "800", "300", "200", "400", "1,600", "2,200", "100", "200", null, "1,000", "1,800", "300", "400", "500", "700", "200", "250", "100", "1,000", "300", "500", "600", "400", "300", "300", "700", "750", "700", "400", "400", "300", "800", "300", "400", "400", "650", "600", "400", "400", "2,100", "300", "350", "250", "300", "100", "250", "300", "150", "1,500", "800", "800", "1,700", "400", "800", "200", "150", "300", "500", "600", "1,700", "1,600", "400", "50", "600", "700", "600", "3,000", "500", "300", "700", "150", "600", "300", "400", "350", "200", "500", "800", "250", "650", "200", "500", "550", "700", "300", "700", "300", "600", "300", "600", "200", "500", "500", "950", "1,800", "200", "300", "100", "300", "800", "600", "1,600", "450", "400", "150", "300", "350", "2,000", "800", "800", "700", "400", "200", "700", "3,000", "200", "200", "300", "1,500", "400", "200", "200", "150", "600", "400", "450", "200", "1,000", "200", "500", "400", "300", "300", "200", "1,000", "3,000", "500", "500", "550", "500", "700", "550", "1,400", "400", "1,100", "600", "400", "950", "300", "600", "900", "250", "700", "300", "1,400", "150", "300", "400", "550", "1,000", "750", "600", "1,500", "200", "300", "500", "1,200", "200", "500", "200", "700", "1,000", "400", "400", "200", "500", "100", "1,300", "1,000", "200", "1,000", "3,000", "200", "1,300", "600", "250", "500", "500", "400", "200", "250", "300", "1,000", "300", "200", "400", "200", "2,500", "600", "150", "600", "200", "150", "600", "150", "300", "150", "1,300", "1,300", "800", "200", "200", "800", "250", "300", "800", "150", "200", "1,200", "550", "400", "1,100", "300", "200", "350", "250", "2,500", "700", "1,500", "1,000", "1,300", "300", "2,400", "500", "300", "500", "150", "350", "1,600", "450", "100", "400", "250", "100", "400", "200", "100", "200", "1,200", "900", "200", "500", "1,000", "500", "350", "250", "250", "300", "400", "350", "800", "200", "250", "550", "200", "600", "600", "600", "2,500", "300", "750", "1,000", "500", "300", "200", "500", "650", "750", "700", "1,400", "300", "300", "1,000", "700", "400", "300", "500", "1,200", "800", "1,350", "400", "500", "550", "250", "250", "100", "500", "1,500", "1,000", "800", "400", "200", "800", "1,500", "500", "400", "800", "100", "400", "400", "550", "400", "400", "300", "400", "500", "500", "300", "500", "300", "300", "1,500", "1,000", "400", "100", "1,200", "800", "250", "200", "400", "1,100", "200", "1,100", "300", "800", "250", "650", "1,900", "700", "600", "1,000", "250", "500", "500", "200", "350", "350", "450", "200", "700", "200", "400", "600", "1,400", "600", "180", "150", "1,300", "200", "350", "300", "300", "250", "300", "800", "800", "900", "200", "1,000", "250", "300", "400", "500", "300", "400", "200", "300", "200", "500", "1,000", "700", "600", "1,500", "1,000", "150", "250", "500", "300", "300", "200", "100", "400", "400", "100", "200", "400", "900", "1,200", "300", "150", "600", "400", "750", "1,500", "200", "500", "500", "500", "1,000", "300", null, "150", "250", "500", "500", "4,000", "350", "200", "500", "350", "450", "600", "1,500", "200", "400", "250", "1,000", "1,000", "500", "500", "500", "500", "800", "700", "300", "500", "3,000", "550", "500", "600", "300", "150", "200", "400", "300", "700", "450", "400", "1,000", "200", "400", "500", "1,100", "500", "500", "400", "550", "300", "1,200", "600", "250", "150", "1,800", "300", "550", "100", "550", "400", "850", "500", "700", "200", "200", "300", "400", "500", "300", "600", "250", "400", "500", "300", "300", "500", "400", "300", "500", "600", "500", "2,100", "500", "3,400", "700", "600", "400", "300", "600", "650", "650", "400", "1,000", "750", "500", "400", "500", "350", "4,000", "800", "400", "200", "600", "300", "1,500", "400", "150", "700", "300", "100", "500", "100", "1,200", "300", "200", "400", "800", "300", "500", "1,700", "500", "200", "400", "250", "200", "350", "2,400", "400", "300", "1,500", "500", "500", null, "350", "250", "350", "150", "250", "350", "500", "300", "400", "2,000", "500", "200", "800", "600", "150", "2,800", "300", "400", "700", "900", "650", "300", "600", "650", "300", "150", "400", "200", "1,700", "1,500", "550", "550", "1,000", "400", "150", "600", "550", "2,000", "300", "150", "1,000", "500", "400", "200", "300", "450", "500", "1,000", "250", "2,000", "3,000", "500", "500", "550", "200", "150", "1,000", "200", "200", "500", "400", "150", "600", "500", "200", "400", "500", "200", "300", "400", "250", "300", "300", "400", "750", "600", "300", "900", "700", "300", "400", "400", "550", "350", "700", "700", "300", "400", "850", "500", "500", "400", "500", "250", "350", "250", "400", "400", "900", "300", "1,100", "500", "250", "1,000", "500", "400", "400", "600", "650", "200", "700", "1,000", "300", "400", "400", "1,000", "750", "150", "1,200", "100", "300", "400", "150", "600", "400", "2,000", "350", "600", "2,200", "450", "400", "250", "600", "200", "400", "600", "500", "250", "300", "550", "400", "400", "250", "650", "300", "300", "450", "400", "600", "500", "350", "350", "450", "150", "500", "700", "100", "200", "800", "200", "80", "1,500", "300", "500", "200", "700", "400", "800", "800", "500", "200", "200", "750", "400", "250", "600", "600", "400", "600", "120", "200", "350", "750", "150", "500", "500", "500", "500", "1,500", "600", "600", "300", "250", "300", "400", "300", "400", "1,050", "300", "450", "500", "250", "1,500", "600", "250", "600", "800", "400", "600", "300", "150", "250", "200", "1,200", "900", "400", "400", "300", "850", "200", "400", "200", "300", "300", "400", "1,000", "600", "700", "1,700", "250", "200", "400", "300", "500", "400", "650", "300", "400", "800", "450", "400", "400", "400", "450", "700", "600", "500", "300", "300", "250", "500", "800", "1,500", "1,000", "150", "600", "500", "300", "600", "1,000", "800", "350", "400", "500", "1,100", "500", "400", "150", "1,200", "550", "300", "800", "900", "250", "300", "600", "100", "300", null, "350", "200", "400", "200", "350", "800", "450", "200", "200", "150", "300", "250", "200", "300", "450", "700", "700", "600", "800", "200", "400", "1,500", "700", "1,000", "400", "1,000", "1,000", "400", "500", "350", "1,300", "300", "300", "300", "300", "350", "1,100", "500", "200", "300", "500", "1,000", "400", "500", "600", "150", "1,300", "200", "1,300", "400", "250", "300", "1,300", "100", "150", "1,000", "700", "800", "450", "400", "550", "700", "700", "400", "300", "150", "350", "300", "400", "500", "300", "600", "1,200", "200", "1,000", "450", "750", "1,000", "300", "300", "1,400", "600", "200", "300", "500", "300", "700", "650", "200", "2,000", "400", "400", "300", "300", "400", "100", "3,400", "200", "200", "800", "200", "300", "1,250", "400", "150", "400", "500", "200", "300", "400", "300", "300", "1,000", "800", "250", "900", "250", "400", "1,000", "800", "1,500", "300", "1,300", "2,500", "350", "400", "500", "150", "1,300", "450", "300", "750", null, "750", "800", "450", "600", "300", "200", "200", "450", "350", "300", "500", "800", "600", "600", "500", "300", "1,200", "200", "350", "300", "150", "200", "300", "150", "500", "300", "150", "300", "400", "650", "1,200", "500", "200", "150", "300", "200", "200", "400", "300", "1,100", "500", "1,000", "700", "250", "100", "1,500", "2,000", "300", "350", "700", "400", "300", "700", "200", "900", "600", "3,000", "400", "500", "500", "600", "500", "1,000", "1,000", "300", "150", "300", "600", "600", "100", "150", "450", "400", "500", "500", "900", "200", "300", "200", "800", "100", "150", "700", "1,500", "600", "450", "850", "700", "250", "1,200", "1,500", "300", "550", "300", "300", "1,200", "900", "400", "500", "550", "500", "200", "700", "500", "400", "100", "300", "450", "400", "400", "200", "500", "300", "300", "250", "550", "500", "500", "300", "400", "300", "600", "600", "1,500", "300", "300", "150", "700", "300", "300", "200", "1,100", "400", "400", "400", "900", "300", "250", "400", "1,200", "500", "200", "1,000", "300", "200", "200", "300", "300", null, "1,400", "800", "500", "300", "400", null, "1,000", "600", "250", "1,200", "700", "400", "250", "300", "1,900", "500", "300", "300", "1,500", "200", "350", "750", "300", "1,100", "300", "400", "250", "200", "450", "400", "600", "1,200", "250", "500", "500", "200", "500", "400", "200", "300", "300", "500", "550", "450", "500", "300", "700", "300", "400", "300", "200", "700", "1,100", "300", "400", "200", "200", "800", "300", "350", "800", "150", "300", "300", "200", "250", "800", "400", "300", "400", "1,400", "1,500", "750", "700", "800", "300", "500", "850", "500", "200", "200", "300", "250", "800", "1,800", "800", "350", "600", "150", "500", "500", "500", "300", "400", "400", "250", "350", "600", "300", "150", "450", "800", "1,000", "500", "800", "350", "800", "300", "150", "1,800", "400", "1,400", "1,200", "1,200", "200", "400", "750", "450", "150", "300", "500", "800", "1,400", "100", "400", "400", "350", "1,000", "400", "500", "150", "600", "2,800", "200", "500", "800", "450", "800", "150", "500", "400", "1,900", "500", "800", "200", "500", "150", "300", "150", "250", "300", "400", "1,300", "400", "800", "300", "400", "200", "600", "1,000", "500", "650", "700", "150", "1,300", "400", "200", "550", "1,600", "150", "600", "200", "500", "700", "600", "400", "250", "1,200", "500", "1,300", "800", null, "450", "250", "200", "1,400", "1,200", "400", "300", "200", "450", "500", "150", "2,000", "300", "200", "850", "1,100", "300", "150", "150", "1,700", "350", "400", "1,200", "150", "4,000", "500", "1,000", "150", "1,000", "300", "350", "250", "200", "750", "1,100", "800", "200", "350", "600", "400", "300", "400", "200", "800", "500", "400", "300", "600", "500", "200", "250", "800", "400", "1,500", "800", "450", "150", "600", "300", "250", "200", "350", "150", "500", "800", "300", "300", "250", "300", "300", "1,000", "300", "200", "400", "400", "300", "250", "600", "600", "150", "400", "1,000", "1,800", "1,000", "1,500", "300", "800", "550", "150", "300", "400", "300", "600", "200", "400", "600", "250", "600", "250", "200", "400", "1,500", "400", "400", "800", "500", "500", "1,100", "800", "200", "700", "3,000", "600", "300", "900", "1,600", "800", "400", "600", "150", "150", "300", "300", "700", "100", "500", "250", "300", "1,000", "400", "300", "400", "500", "300", "400", "600", "300", "700", "200", "250", "500", "700", "200", "200", "300", "300", "300", "700", "200", "800", "400", "700", "400", "300", "700", "400", "200", "1,300", "200", "1,000", "300", "250", "1,800", "400", "350", "500", "150", "200", "400", "200", "200", "300", "800", "1,300", "400", "500", "400", "600", "400", "200", "1,200", "200", "300", "400", "500", "400", "1,600", "500", "500", "600", "300", "300", "400", "300", "800", "400", "250", "850", "400", "300", "150", "150", "300", "300", "2,400", "150", "500", "250", "300", "1,000", "700", "600", "600", "500", "200", "500", "400", "250", "700", "450", "250", "300", null, "100", "200", "250", "300", "200", "500", "150", "400", "600", "550", "500", "200", "3,500", "200", "200", "800", "750", "400", "300", "400", "300", "300", "600", "1,500", "500", "150", "300", "300", "400", "300", "600", "300", "250", "1,000", "300", "300", "500", "150", "200", "150", "600", "700", "400", "150", "1,400", "400", "200", "800", "300", "100", "150", "250", "2,200", "1,600", "600", "400", "600", "550", "150", "300", "350", "250", "350", "350", "300", "800", "800", "300", "600", "500", "250", "250", "400", "1,300", "350", "150", "100", "1,000", "350", "800", "400", "400", "200", "350", "1,200", "200", "600", null, "400", "1,500", "500", "1,500", "150", "400", "700", "200", "150", "400", "450", "650", "600", "600", "300", "1,500", "600", "150", "450", "3,400", "600", "400", "100", "1,100", "600", "500", "300", "600", "400", "1,500", "200", "400", "600", "500", "600", "400", "300", "500", "800", "300", "400", "1,800", "450", "450", "1,000", "1,100", "600", "600", "750", "300", "200", "350", "350", "400", "600", "1,800", "800", "400", "1,000", "250", "400", "350", "150", "600", "100", "600", "150", "600", "3,000", "300", "250", "500", "400", "400", "500", "300", "300", "1,400", "700", "300", "300", "1,200", "1,400", "250", "200", "500", "250", "250", "200", "250", "800", "550", "800", "600", "300", "200", "500", "1,200", "250", "1,700", "600", "400", "500", "1,000", "250", "250", "200", "350", "80", "300", "200", "150", "550", "350", "1,000", "200", "300", "350", "750", "600", "300", "550", "200", "200", "300", "200", "500", "750", "300", "550", "200", "500", "100", "500", "700", "1,200", "350", "300", "250", "150", "300", "1,500", "400", "400", "600", "1,300", "550", "300", "1,300", "500", "400", "500", "950", "650", "400", "200", "3,000", "600", "1,500", "300", "150", "500", "300", "1,500", "1,000", "500", "1,200", "700", "400", "1,000", "1,100", "300", "250", "300", "600", "300", "750", "300", "350", "650", "350", "1,000", "550", "1,500", "550", "300", "750", "300", "250", "500", "600", "500", "1,800", "150", "500", "300", "400", "200", "250", "500", "350", "1,100", "350", "350", "350", "400", "400", "300", "250", "500", "300", null, "400", "500", "800", "300", "300", "300", "1,000", "500", "1,200", "500", "500", "250", "650", "500", "300", "300", "200", "500", "800", "600", "500", "200", "200", "250", "230", "400", "900", "500", "550", "300", "400", "250", "1,100", "1,200", null, "450", "500", "500", "400", "400", "200", "1,500", "800", "600", "1,600", "300", "200", "350", "1,000", "600", "600", "600", "250", "300", "800", "350", "250", "500", "500", "2,500", "150", "500", "500", "600", "400", "1,500", "350", "1,500", "250", "1,600", "400", "350", "400", "300", "1,600", "600", "700", "500", "900", "400", "200", "250", "300", "700", "300", null, "600", "750", "200", "200", "600", "300", "350", "400", "1,300", "700", "250", "500", "650", "600", "1,600", "600", "700", "300", "200", "1,000", "300", "800", "1,200", "900", "1,500", "650", "250", null, "600", "300", "800", "400", "500", "550", "250", "250", "400", "1,300", "550", "800", "300", "400", "700", "350", "600", "600", "600", "400", "600", "300", "250", "150", "600", "150", "400", "450", "450", "400", "1,800", "1,000", "800", "400", "250", "300", "600", "400", "550", "250", "1,700", "300", "200", "750", "180", "500", "400", null, "200", "500", "350", "150", "300", "300", "350", "300", "450", "350", "400", "800", "300", "2,500", "400", "200", "700", "3,000", "300", "300", "300", "700", "400", "700", "200", "800", "600", "550", "600", "400", "200", "900", "500", "350", "400", "750", "600", "600", "700", "1,200", "1,700", "300", "150"], "marker": {"color": "rgb(255, 178, 102)", "line": {"color": "DarkSlateGrey", "width": 1}, "size": 10}, "mode": "markers", "type": "scatter", "x": [null, 3.1, 4.0, 4.2, 3.9, 4.1, 3.3, 4.4, 3.9, 3.8, 3.3, null, 3.1, 3.5, 3.8, 2.9, 4.1, null, 3.9, 4.1, 3.8, null, 4.4, null, 4.1, 3.7, null, 3.5, null, 3.4, 3.7, 4.5, null, 4.3, 3.6, 2.8, 3.6, 3.8, 4.0, 3.8, 3.3, 3.4, 3.5, 4.4, null, 4.3, 3.2, 4.0, 2.9, null, 3.1, 4.0, 3.0, 3.4, 4.4, null, 3.9, 4.4, null, 3.2, null, 4.1, 4.6, 3.6, 3.6, 3.9, 3.4, 4.1, null, 3.5, 3.7, 3.7, 3.9, null, 3.9, 4.0, 3.8, 2.9, 3.9, 3.8, 2.8, 3.6, 2.8, 4.0, 3.7, 3.8, null, null, 3.1, 2.9, 3.9, 3.2, 4.0, 3.7, 3.0, null, 4.8, null, null, 3.6, 3.6, null, 3.3, 3.7, null, 4.1, 3.8, null, 4.0, 3.8, 4.0, 3.5, null, 4.0, 3.6, 2.9, 4.5, 4.4, 2.8, 3.6, 3.8, 4.4, 4.1, 4.3, 3.8, 3.7, 3.7, null, null, null, 2.9, 3.4, 3.5, 3.0, 3.8, 3.9, 3.5, 3.1, 3.0, null, 3.8, null, 3.1, 4.0, 3.8, 3.4, 3.7, 3.4, 3.0, 4.4, 4.1, 4.4, 4.0, null, null, 3.3, 3.8, 2.9, 4.2, null, 3.0, null, null, null, 4.1, 3.9, 3.6, 3.8, 2.1, 3.6, null, 4.2, 3.8, 3.9, null, null, 3.5, null, 3.2, 3.5, 3.6, null, 4.4, 3.6, 4.2, null, 4.1, 3.6, 4.0, null, 3.8, 4.1, 4.4, 3.7, 3.1, 4.0, 3.7, 3.7, 3.4, 4.3, 3.3, 4.4, null, 4.1, 3.7, 3.4, 3.6, null, 4.1, 3.6, 3.8, 4.2, null, null, 4.1, 3.1, 4.4, 3.1, null, null, 3.9, null, 3.9, null, null, 3.0, 3.8, 4.0, 4.0, 4.9, 3.0, 4.0, 3.9, 3.8, 4.0, 4.3, 3.9, 4.2, 4.0, null, 4.0, 4.1, null, 3.7, 4.1, 4.2, 3.5, 3.3, 3.9, 3.9, 3.9, null, 3.1, null, 4.3, 4.2, null, 3.7, 4.1, null, 3.3, 3.0, 3.2, 3.4, 4.3, 4.3, 3.9, 3.6, null, 4.0, 3.5, null, 3.9, 3.4, 3.7, 4.1, 3.4, 3.6, 3.1, null, 3.4, 3.9, 3.7, 4.1, 3.7, 4.2, 2.8, 4.3, 3.8, 4.0, null, null, 3.7, 3.5, 3.7, 4.3, 3.6, 4.0, 4.3, null, null, 3.5, null, 3.8, 3.3, 4.0, 3.8, 3.5, 3.9, 4.1, 3.8, 3.4, 3.9, 3.8, 3.8, 4.4, 3.2, 3.7, 3.8, 3.6, 4.1, 3.3, 3.5, 3.3, 3.7, 4.4, 3.6, 3.7, 2.7, null, 3.5, 3.7, 4.0, 3.7, 4.2, 3.7, 4.2, null, 3.5, 3.6, 3.5, 4.1, 4.0, 4.0, 4.3, 3.8, 3.8, null, 3.8, 3.7, null, 3.7, 3.5, 3.6, 4.0, 2.8, 4.1, 3.7, 3.9, 3.5, 4.1, null, 4.2, 4.0, null, 3.7, 4.3, 4.0, 4.0, 3.7, 3.5, 3.6, 4.0, 4.4, 3.9, null, 3.2, 4.2, 4.1, 4.3, null, 3.3, 4.1, 4.1, null, null, 3.5, 4.2, 3.6, 4.1, 3.9, 4.0, null, null, 3.9, 2.8, 4.1, 3.5, null, null, 3.8, 3.5, 4.5, 3.6, 3.2, null, 3.9, null, 3.5, 3.3, 3.9, 3.0, 4.3, 3.9, 3.8, 3.5, null, 3.4, null, 3.5, null, 3.7, 4.1, 3.8, 3.2, 2.8, 3.5, 3.3, null, null, 3.4, 4.0, null, 3.5, 3.9, 3.7, 3.8, 4.0, 3.2, 4.2, 4.3, 3.7, 4.0, 2.8, 3.3, 3.4, 3.3, 3.7, 3.8, 3.7, 3.3, 3.7, null, 4.1, 4.2, 4.0, 4.0, 3.6, null, 3.7, 4.2, 3.2, 3.8, 3.5, 3.3, 3.8, null, null, null, 3.4, 3.8, 3.8, 4.1, 3.2, 3.8, 4.0, 3.7, 3.0, 3.2, 4.4, null, 3.3, 3.2, 4.0, 3.7, null, 4.1, 2.6, null, 4.1, 4.1, null, 3.6, 3.8, 2.8, null, 2.9, 4.0, 3.0, 3.7, 4.3, null, 3.9, 3.5, 4.2, 4.0, null, 3.6, 3.5, 3.7, 3.8, 3.4, 4.2, 3.6, 4.5, 4.1, 3.7, 3.9, 3.7, 4.5, 3.6, 4.0, null, 3.7, null, 4.0, 3.9, 4.1, 4.2, null, 3.4, 3.3, null, null, 2.7, 3.2, null, 3.4, 3.7, null, 4.0, 3.4, null, 3.7, 2.7, null, 4.3, null, 4.3, 3.7, 3.5, 4.3, null, 3.6, 2.2, 4.2, 3.8, null, 4.3, 4.1, 3.1, 3.1, 3.9, 4.4, 3.9, 3.5, 3.4, 2.8, 3.8, null, 3.8, null, 3.0, 4.2, null, 2.9, 4.2, 4.4, null, 3.3, 3.3, 4.0, 3.9, null, 4.2, 3.2, null, 3.5, 3.1, null, null, 4.1, null, 3.8, 4.6, null, null, 4.2, 3.2, 3.5, 3.7, 3.8, 3.8, 3.4, 4.0, 3.3, 3.2, 3.7, 3.5, null, 3.8, 3.5, 3.5, 4.3, null, 3.9, 3.8, 4.1, 4.2, 3.6, 3.9, 3.9, 3.1, 3.5, 3.7, null, 4.2, 4.5, 3.7, null, 4.0, null, 3.2, 4.1, 4.1, 4.3, 3.4, 4.0, 4.1, 3.5, 3.8, 3.6, 3.9, 2.7, 4.3, 4.0, 3.8, 4.1, 3.9, 3.5, 3.6, 3.9, 3.5, 4.1, 3.6, 3.2, 3.2, 3.7, null, null, 3.4, 3.6, null, 4.3, 2.4, 3.5, 3.3, 3.7, 3.2, null, 3.0, 4.0, 4.0, 3.9, 3.5, 3.9, 3.4, 3.7, null, 4.0, 3.1, null, 3.2, 3.6, 3.6, 3.7, 4.1, 3.7, 3.6, 4.3, 2.9, 4.2, 3.8, null, 4.4, 3.5, 4.0, 3.3, 3.7, 3.8, 3.4, 4.0, 3.5, null, 2.9, 3.3, 3.7, 3.9, 3.6, 3.9, 3.3, 3.4, 3.7, 4.4, 2.8, 3.5, 3.9, 2.9, null, 4.0, null, 3.8, 2.8, 4.3, 3.6, 2.8, 3.5, 3.2, 3.8, 3.3, null, 2.9, 4.2, 2.7, null, 3.4, null, 3.7, 4.1, null, 4.4, 2.7, null, 3.6, 3.1, null, 4.3, 2.8, 3.8, 3.5, 3.2, 3.6, 3.9, 3.9, 3.6, 3.9, 4.1, null, null, 4.5, 3.3, 3.8, null, null, 3.3, 3.6, 3.5, 3.3, 4.1, null, 4.2, 3.9, 3.7, 3.9, 3.8, 3.6, 3.9, 4.4, 3.7, 3.5, 4.0, null, 3.8, 4.0, 2.9, 3.9, 3.9, 4.1, 3.7, 3.5, 3.1, null, 4.4, 3.7, 2.9, 2.6, 4.0, 2.5, 3.6, 3.3, 3.8, 3.7, 3.5, 2.8, 3.9, 3.8, 3.7, 3.0, null, 3.2, null, 4.4, 3.4, 3.7, 3.5, null, 4.3, 4.3, null, 3.4, 3.3, 2.8, 3.4, 4.2, 3.9, 3.8, 4.4, null, 4.1, 3.6, 3.4, 3.0, 3.7, 2.1, 3.7, 3.3, 3.9, null, 3.0, null, 3.0, 3.1, 2.8, 3.5, 3.4, 3.3, 4.1, null, 3.9, null, 4.1, 3.7, 3.8, 3.8, 3.0, null, 3.9, 3.3, 4.4, 3.5, 3.7, 4.3, 3.7, 4.2, 3.6, null, 4.1, null, 4.0, 3.4, 2.7, 2.9, null, 4.4, 3.6, 4.0, null, 3.1, 3.4, null, 3.3, 3.8, 3.4, 3.4, null, 4.2, 4.1, null, null, null, 3.2, 3.4, 3.9, 4.1, 2.6, 3.9, 3.0, 3.6, 4.0, 3.7, 4.3, null, null, 4.1, 3.4, 4.3, 3.6, 4.5, 3.9, 3.8, 4.1, 3.6, 3.3, 3.7, 3.8, 3.7, 3.9, 3.6, 4.2, null, 4.1, 3.9, null, 4.2, 3.8, 4.4, 3.7, null, null, 3.9, 2.9, 3.3, 3.1, 3.8, 4.4, 3.2, 3.6, 4.3, 4.1, 3.8, 2.9, 4.1, 3.1, 3.4, null, 3.9, 3.7, null, 4.0, 2.8, 3.3, 3.8, null, 4.0, 2.2, 4.0, null, 4.2, 3.4, 3.3, 3.6, 3.3, 3.6, 3.8, 2.8, 3.6, 3.8, null, null, 3.5, 4.0, null, null, 4.3, 3.3, 3.4, 3.8, 3.7, 3.6, 4.5, null, 3.0, 3.1, 3.6, 3.7, 3.7, null, 4.0, null, 4.2, null, null, null, 3.7, 3.2, 4.3, 4.5, null, 3.2, 3.5, 3.9, 3.8, 3.5, 4.1, 3.1, 4.2, null, 3.5, 4.4, 3.8, 4.0, null, 3.1, 3.8, null, 3.9, null, 3.9, 3.8, null, null, 3.6, 3.5, 4.2, 3.9, null, 1.8, 4.0, 4.0, null, 4.1, null, 3.5, 4.1, 2.6, null, 3.7, 3.5, null, 3.8, 4.5, null, 3.3, 3.2, 2.9, null, null, 4.2, 3.7, 3.9, 3.8, 4.0, 3.9, 3.1, 4.3, 4.3, 4.7, 3.8, 2.9, 4.1, null, 4.1, 4.2, null, 3.8, 4.4, 3.8, 3.7, 3.8, 4.0, null, 2.8, null, 4.4, 3.7, null, 3.4, 4.0, 4.1, null, null, 3.7, 2.9, null, 3.1, 4.4, 3.3, 3.3, 4.0, 3.8, 3.6, null, 4.4, 4.6, 4.3, 3.3, 3.8, 4.0, null, 4.1, 3.9, 3.8, 3.7, 3.3, 4.2, 4.2, 2.6, null, 3.5, 3.1, 2.8, null, 4.1, 4.0, 3.1, 3.4, 3.5, 3.4, 4.3, 3.7, null, null, null, 3.9, null, 4.3, 3.9, 4.1, 3.8, 4.3, 2.6, 3.6, 3.4, 4.2, null, 4.0, null, 3.7, 3.6, 3.6, 3.6, 4.4, 3.5, 4.4, 3.5, 4.5, null, null, 3.6, 3.9, 3.1, null, 4.4, 3.4, null, 3.9, 3.8, 3.7, null, 4.1, 3.4, 3.7, 4.0, 3.5, 4.2, 4.4, 3.5, null, 3.9, 3.2, 4.0, 3.5, 3.4, 4.4, 3.7, 4.0, null, 4.1, null, 2.8, 3.9, 3.8, 4.0, 2.8, 3.7, 3.5, null, 2.3, null, null, 4.1, 3.7, 3.7, 2.8, 3.3, 3.0, null, null, 3.6, 4.2, 3.8, 3.1, 3.0, 4.2, 3.1, 3.3, 3.7, null, null, 3.8, 4.1, 3.9, null, 4.1, null, 3.7, 3.5, 3.7, 3.7, 3.8, 3.5, 2.8, 3.9, 3.2, 3.2, 4.0, 3.1, 3.9, null, 4.2, 4.4, 4.4, 2.8, 4.2, 2.8, 3.1, 3.7, 2.8, 4.0, 3.8, 3.6, 2.9, 3.8, 4.2, 4.5, 3.6, null, null, null, 3.7, null, 3.2, 3.4, null, 3.2, 4.3, 3.8, null, 3.7, 3.4, 4.0, 4.0, 3.4, 2.5, 3.3, 3.8, 4.1, 3.4, 4.5, 3.8, 4.6, 3.1, 4.4, 3.2, 2.7, 3.7, 3.6, 3.4, null, 2.9, null, 4.0, 3.2, null, 4.0, 3.8, 4.2, 3.5, 2.8, null, 3.8, 4.6, null, 3.4, 3.0, null, null, null, null, 4.1, 4.5, 4.2, 3.5, 3.4, null, null, null, 3.0, 3.3, 3.2, 3.9, 3.6, null, 4.2, 4.0, 2.9, null, 3.4, 3.4, null, 3.8, 3.8, 3.7, 4.0, 3.6, 4.2, 3.7, 4.5, 4.1, 3.2, null, 4.0, 3.0, 4.0, 3.2, 2.9, 4.3, 4.0, 4.7, null, 4.0, 3.2, 4.0, 3.9, 4.1, 3.3, 3.7, null, null, 3.8, null, 3.1, 4.0, 3.6, null, 3.6, 3.8, 3.4, 3.3, 3.2, 4.5, null, null, 4.1, 3.6, 4.2, 3.0, null, null, 4.3, 4.1, 2.8, 3.6, 3.1, 3.8, 4.0, 4.1, null, 4.2, 4.1, 3.7, 2.6, 3.2, 3.3, 2.8, 3.9, null, 3.9, 3.1, 3.8, null, 3.1, 4.3, null, 4.3, 3.6, 3.9, null, 3.2, 3.6, null, 3.6, null, 4.1, 4.0, 4.3, 3.3, 3.3, null, null, 3.2, 3.5, 3.6, 3.5, 2.9, 3.9, 3.9, 3.4, 4.2, 3.8, 3.4, 3.6, 4.0, 4.1, 3.6, 4.2, 3.7, 2.7, 3.7, 3.3, 3.1, null, 2.8, 3.9, 4.3, null, 3.3, 3.7, null, 3.7, null, 3.0, 4.4, null, 3.4, 4.1, 3.6, 3.1, 4.0, 4.1, null, 3.6, 4.4, 2.9, 3.8, 4.0, 3.6, 3.6, 3.7, 3.8, 3.6, null, 3.5, null, 4.1, null, null, null, null, 4.3, 3.2, 3.3, 3.4, 3.9, 4.1, null, 3.0, null, null, 3.2, 3.4, 4.1, 4.0, null, 3.7, 3.3, 3.3, null, 4.0, null, 3.1, 3.2, 3.8, 3.2, 3.9, 4.3, 3.4, 3.7, 3.8, 3.2, 3.9, 3.2, null, 4.4, 3.3, 3.6, 3.3, null, 3.6, null, 3.9, null, 4.0, 3.8, 3.9, null, 3.8, null, 4.1, 4.2, 3.5, 3.8, null, null, 3.9, 3.6, 3.7, 4.5, 3.9, 3.2, 3.5, 3.3, 3.8, 3.1, 3.3, 2.3, 3.3, 3.7, null, 2.6, 3.4, 3.1, null, 3.6, 3.8, 4.0, 3.8, null, 3.2, null, null, 3.8, 3.6, 3.7, 3.6, 3.6, 4.0, 3.5, 3.8, 3.6, 3.6, 4.1, 4.1, null, 3.3, 3.1, null, 3.1, 3.6, 4.0, 3.4, 3.7, 3.9, 3.3, null, 3.6, 4.3, 3.2, 3.8, 3.7, 4.2, 3.3, 3.9, 3.8, 3.6, 3.6, 3.8, 4.1, 3.8, 4.0, null, 3.6, 3.9, null, 2.7, 4.4, 3.2, 3.8, 3.7, null, 2.6, null, 4.1, 3.7, 3.6, 3.5, null, null, 3.3, 3.0, 3.9, 3.6, 3.6, 3.2, null, 3.9, 3.5, 4.3, 3.2, null, 3.6, 3.2, 2.9, 3.6, 3.8, null, 3.2, 3.8, 4.1, null, 4.2, null, 3.3, 3.7, 3.6, null, null, 4.3, 3.6, 3.1, 3.9, 4.1, 4.1, 4.2, 3.9, 4.1, 4.0, 3.4, 4.5, null, 3.9, 3.4, 3.3, 4.1, 3.5, 4.0, null, 4.2, 2.6, null, 3.5, 3.5, 2.9, 3.8, 3.6, 3.7, 3.4, 4.6, 3.6, 3.8, 3.5, 4.3, 3.5, 3.1, 3.7, null, null, 4.3, 4.4, 3.9, 3.7, 4.1, 3.7, 3.5, 3.8, null, 3.0, null, 3.5, 3.0, 4.2, 3.2, 3.3, 3.2, 3.8, 3.5, 3.7, 4.5, 3.7, 2.9, 3.7, 3.6, 3.1, 3.2, null, null, 3.6, 3.5, 3.4, null, 4.1, 3.4, null, 3.3, 4.4, 3.4, null, 4.1, 3.4, 3.7, 4.3, 3.4, 3.3, 3.5, null, 3.5, 3.9, 4.0, 3.7, 4.4, 3.7, 3.3, 4.1, 2.8, 3.8, 3.9, 4.5, 3.3, null, 3.3, null, 3.7, 4.3, 3.9, 3.9, null, 3.1, null, 2.7, 3.7, 3.3, 4.3, 4.0, 3.2, null, 4.2, 4.3, 3.4, null, 3.8, null, null, 3.6, 4.3, 3.2, null, 3.5, 3.6, null, 2.8, 3.1, 2.9, 3.7, 3.6, null, 3.9, 3.5, 3.0, 3.8, 4.3, 3.4, 4.0, 3.9, 3.9, null, 3.6, 3.6, 4.2, null, null, 3.8, 3.0, 2.9, 4.3, 3.6, 3.7, 3.9, 4.5, 3.4, null, 4.2, 2.8, 4.2, null, 4.1, 3.0, 3.0, 4.6, 4.3, 3.3, 4.2, 3.6, null, 3.7, 3.0, 4.4, 4.1, 3.6, 4.2, 4.1, 3.3, 4.2, 4.1, null, 3.7, 3.8, 4.1, 3.9, 4.2, 4.1, 4.0, null, 3.6, 4.0, 3.5, 4.2, 3.9, 3.3, 4.0, 3.7, null, 3.3, 3.6, 3.9, 3.5, 3.6, 3.4, 3.4, 3.5, 3.3, 4.0, 3.7, 3.6, 4.3, 3.3, 2.8, 3.7, 3.8, 2.9, 4.2, 3.2, 4.0, null, 3.8, 3.9, null, 4.5, 3.9, 3.8, 3.3, 4.2, 4.2, 3.4, 3.9, 3.1, 3.8, 3.5, null, 3.0, 3.3, null, 3.7, 4.0, 3.6, 3.0, 3.6, null, 3.5, 4.0, null, 3.3, 3.7, 2.8, 3.4, 3.1, null, 4.0, 4.4, 3.8, 3.9, null, 2.8, 3.4, 3.8, 3.7, 4.2, 4.0, 3.8, 4.6, 3.1, 3.7, 4.0, 2.6, 3.5, 4.3, 3.9, 3.5, 3.5, 3.9, 3.5, null, 3.1, 4.2, 4.1, 3.8, 3.2, 3.3, 3.9, 3.4, 4.2, 3.7, 4.3, 2.9, 3.8, 3.8, 3.7, 4.1, 4.0, 4.3, 2.5, 4.3, 3.8, 3.1, 3.9, null, null, 3.5, 3.7, 2.9, 4.2, 4.4, 3.9, null, 3.9, 4.1, 3.4, null, 3.6, 3.9, 3.9, null, 3.2, 4.1, 3.2, 4.5, 3.5, 3.7, 3.1, 3.0, 4.3, null, 3.9, 4.3, 3.7, 3.9, 4.0, null, 3.9, 2.8, 4.0, 3.5, 4.0, 3.5, 3.4, 3.1, null, 3.1, 3.9, 4.3, 3.9, 3.8, 3.4, 4.2, null, 4.1, 4.2, 3.8, 3.1, 4.2, 3.7, 3.6, 3.0, 3.5, 3.4, 3.6, 3.6, 3.0, 4.2, 3.8, 2.2, 4.2, 3.5, 3.2, 4.1, 4.1, 2.9, 4.1, 3.8, 4.2, 3.8, 4.0, 3.8, 4.3, 3.8, null, 3.8, null, 3.7, null, null, 3.3, null, 3.8, 3.6, 2.9, 3.8, 3.4, 4.1, null, 4.0, 4.4, 3.9, 4.1, 4.4, 3.1, 3.6, null, 3.9, 3.1, 4.3, 3.5, 4.4, 2.9, 3.1, 4.1, 3.3, null, 3.8, null, 3.9, null, 3.8, 4.1, 4.1, 3.8, 4.0, 4.3, null, 3.2], "y": [0, 21, 131, 3236, 225, 402, 9, 712, 64, 46, 184, 0, 7, 13, 291, 89, 289, 0, 1214, 207, 433, 0, 280, 0, 121, 203, 0, 44, 0, 9, 214, 3468, 0, 114, 163, 94, 68, 263, 166, 208, 11, 7, 14, 1084, 0, 1258, 4, 99, 9, 0, 94, 960, 12, 19, 2487, 0, 100, 1026, 0, 47, 0, 2049, 1439, 10, 34, 237, 5, 604, 0, 18, 86, 833, 57, 0, 15, 370, 72, 75, 185, 89, 11, 96, 152, 894, 12, 171, 0, 0, 42, 23, 155, 22, 787, 74, 13, 0, 498, 0, 0, 14, 53, 0, 10, 49, 0, 782, 434, 0, 167, 29, 210, 21, 0, 51, 51, 18, 1426, 3870, 189, 11, 43, 103, 452, 165, 18, 29, 69, 0, 0, 0, 76, 46, 6, 241, 178, 181, 888, 82, 241, 0, 131, 0, 41, 40, 33, 9, 106, 11, 88, 2634, 719, 861, 89, 0, 0, 6, 285, 67, 3230, 0, 57, 0, 0, 0, 117, 135, 17, 503, 242, 18, 0, 484, 82, 49, 0, 0, 15, 0, 14, 8, 136, 0, 480, 21, 337, 0, 616, 218, 269, 0, 110, 118, 375, 23, 31, 142, 79, 21, 334, 476, 5, 1154, 0, 1225, 39, 130, 27, 0, 101, 32, 132, 7330, 0, 0, 131, 9, 1808, 53, 0, 0, 28, 0, 410, 0, 0, 28, 32, 324, 347, 201, 20, 34, 327, 41, 592, 791, 154, 822, 195, 0, 942, 1847, 0, 23, 163, 289, 12, 16, 179, 235, 59, 0, 60, 0, 597, 240, 0, 47, 59, 0, 8, 21, 9, 12, 994, 994, 421, 12, 0, 30, 11, 0, 238, 36, 47, 759, 27, 17, 16, 0, 14, 57, 21, 168, 420, 217, 253, 1708, 120, 97, 0, 0, 33, 12, 53, 1048, 108, 18, 450, 0, 0, 19, 0, 47, 11, 104, 38, 11, 355, 232, 632, 23, 48, 77, 189, 638, 38, 111, 41, 8, 183, 5, 8, 5, 85, 523, 27, 291, 402, 0, 14, 93, 233, 97, 800, 39, 2291, 0, 7, 27, 9, 41, 59, 138, 548, 23, 151, 0, 44, 24, 0, 73, 10, 10, 48, 11, 552, 38, 27, 131, 971, 0, 289, 494, 0, 176, 74, 181, 49, 229, 17, 519, 76, 1053, 152, 0, 11, 185, 1241, 1269, 0, 5, 1543, 889, 0, 0, 17, 1289, 13, 559, 49, 66, 0, 0, 100, 178, 272, 10, 0, 0, 427, 8, 504, 15, 5, 0, 487, 0, 42, 67, 89, 148, 324, 34, 170, 11, 0, 9, 0, 9, 0, 66, 138, 38, 4, 253, 11, 4, 0, 0, 8, 212, 0, 58, 110, 24, 71, 1229, 294, 136, 790, 129, 57, 185, 21, 27, 4, 35, 60, 50, 7, 16, 0, 553, 570, 89, 100, 196, 0, 132, 1461, 7, 261, 13, 12, 197, 0, 0, 0, 10, 759, 18, 270, 13, 278, 282, 28, 290, 86, 385, 0, 57, 4, 507, 399, 0, 118, 56, 0, 531, 108, 0, 32, 621, 20, 0, 13, 109, 9, 17, 3621, 0, 450, 25, 1077, 776, 0, 64, 20, 837, 110, 6, 403, 91, 155, 1792, 128, 0, 66, 7854, 23, 1087, 0, 24, 0, 499, 50, 448, 226, 0, 9, 5, 0, 0, 53, 70, 0, 10, 14, 0, 57, 6, 0, 88, 73, 0, 3592, 0, 429, 41, 41, 345, 0, 91, 479, 1703, 38, 0, 972, 559, 85, 10, 35, 571, 728, 33, 46, 63, 33, 0, 37, 0, 11, 233, 0, 67, 444, 2055, 0, 4, 7, 605, 55, 0, 476, 4, 0, 6, 28, 0, 0, 100, 0, 12, 1095, 0, 0, 3236, 5, 8, 48, 48, 94, 4, 243, 59, 17, 42, 436, 0, 80, 19, 43, 165, 0, 164, 2332, 125, 676, 26, 448, 247, 11, 14, 92, 0, 1345, 727, 115, 0, 362, 0, 4, 701, 364, 446, 48, 287, 1750, 435, 83, 101, 46, 182, 290, 508, 64, 28, 48, 433, 16, 45, 31, 110, 57, 6, 4, 47, 0, 0, 6, 199, 0, 512, 392, 7, 9, 33, 7, 0, 7, 147, 783, 53, 24, 885, 19, 19, 0, 53, 68, 0, 20, 104, 59, 31, 289, 24, 199, 189, 539, 58, 34, 0, 290, 45, 553, 6, 96, 187, 61, 2164, 11, 0, 27, 14, 69, 56, 13, 914, 110, 4, 19, 192, 148, 16, 1214, 4, 0, 113, 0, 182, 26, 225, 23, 92, 29, 4, 156, 10, 0, 28, 53, 53, 0, 276, 0, 50, 71, 0, 180, 48, 0, 10, 151, 0, 3468, 25, 32, 229, 6, 25, 281, 468, 29, 18, 3238, 0, 0, 3163, 9, 22, 0, 0, 4, 8, 185, 17, 18, 0, 800, 130, 21, 1049, 120, 19, 145, 109, 20, 283, 1310, 0, 56, 220, 539, 1142, 747, 152, 106, 52, 8, 0, 635, 36, 18, 254, 61, 157, 61, 9, 152, 168, 12, 34, 420, 37, 514, 8, 0, 4, 0, 136, 58, 15, 13, 0, 294, 251, 0, 17, 6, 137, 5, 984, 718, 122, 725, 0, 51, 19, 13, 88, 242, 479, 51, 4, 463, 0, 39, 0, 13, 106, 420, 140, 10, 4, 197, 0, 326, 0, 786, 68, 22, 146, 19, 0, 462, 7, 203, 53, 60, 171, 96, 142, 42, 0, 1320, 0, 76, 153, 442, 6, 0, 2662, 10, 200, 0, 4, 8, 0, 7, 34, 11, 76, 0, 337, 42, 0, 0, 0, 5, 78, 510, 399, 283, 118, 36, 62, 48, 21, 549, 0, 0, 266, 5, 168, 15, 3486, 18, 31, 520, 16, 7, 19, 71, 19, 84, 12, 2714, 0, 1156, 125, 0, 1177, 66, 3712, 37, 0, 0, 96, 33, 16, 113, 92, 1972, 27, 42, 1187, 1359, 69, 57, 159, 10, 32, 0, 55, 24, 0, 55, 56, 44, 61, 0, 783, 406, 126, 0, 1745, 24, 4, 21, 34, 94, 591, 70, 11, 46, 0, 0, 13, 253, 0, 0, 620, 5, 7, 29, 32, 10, 236, 0, 16, 5, 195, 19, 17, 0, 332, 0, 984, 0, 0, 2508, 47, 4, 1370, 819, 0, 4, 10, 72, 25, 14, 38, 7, 1175, 0, 13, 4884, 500, 25, 0, 23, 362, 0, 331, 0, 536, 251, 0, 0, 31, 70, 1647, 250, 0, 225, 87, 65, 0, 558, 0, 11, 29, 100, 0, 47, 178, 0, 22, 2073, 0, 9, 5, 25, 0, 0, 70, 102, 951, 67, 568, 485, 9, 241, 456, 277, 464, 89, 2450, 0, 45, 166, 0, 273, 1106, 623, 220, 229, 155, 0, 38, 0, 2389, 19, 0, 11, 1871, 500, 4, 0, 257, 48, 0, 30, 949, 7, 6, 113, 111, 16, 0, 100, 4694, 476, 7, 21, 386, 0, 1858, 109, 36, 50, 4, 34, 2720, 60, 0, 177, 17, 110, 0, 383, 372, 11, 17, 9, 4, 3624, 130, 0, 0, 0, 184, 0, 1187, 357, 511, 12, 450, 158, 12, 9, 133, 0, 76, 0, 111, 26, 79, 35, 1856, 15, 790, 15, 1878, 0, 0, 34, 680, 7, 0, 191, 12, 0, 39, 46, 63, 0, 111, 83, 44, 130, 221, 3236, 1804, 21, 0, 79, 13, 147, 11, 6, 488, 30, 410, 0, 970, 0, 19, 236, 18, 2852, 137, 434, 10, 0, 176, 0, 0, 1858, 73, 34, 34, 4, 289, 0, 0, 34, 819, 109, 14, 18, 505, 9, 12, 17, 0, 0, 19, 115, 123, 0, 360, 0, 468, 180, 34, 47, 311, 13, 14, 924, 16, 7, 863, 19, 191, 0, 1223, 118, 2867, 261, 1431, 211, 7, 323, 302, 25, 37, 14, 54, 194, 860, 3843, 159, 0, 0, 0, 28, 0, 7, 6, 0, 64, 995, 63, 0, 31, 8, 776, 417, 16, 47, 6, 149, 59, 15, 2198, 16, 866, 122, 2041, 4, 82, 126, 192, 5, 0, 8, 0, 419, 5, 0, 789, 14, 2032, 10, 38, 0, 39, 2332, 0, 7, 98, 0, 0, 0, 0, 167, 1508, 432, 185, 9, 0, 0, 0, 16, 4, 4, 185, 180, 0, 175, 76, 79, 0, 16, 215, 0, 208, 69, 67, 790, 211, 38, 39, 3991, 81, 4, 0, 39, 197, 74, 22, 166, 2577, 233, 4811, 0, 213, 835, 132, 267, 853, 4, 195, 0, 0, 109, 0, 6, 1203, 42, 0, 24, 94, 6, 11, 6, 1238, 0, 0, 1785, 23, 275, 20, 0, 0, 108, 337, 92, 24, 117, 70, 169, 129, 0, 376, 61, 33, 72, 17, 4, 100, 681, 0, 82, 166, 22, 0, 8, 2304, 0, 1196, 70, 247, 0, 21, 140, 0, 11, 0, 41, 151, 2447, 4, 12, 0, 0, 20, 7, 38, 21, 142, 54, 783, 31, 239, 44, 9, 154, 610, 109, 9, 1920, 15, 57, 50, 7, 245, 0, 19, 215, 84, 0, 4, 22, 0, 14, 0, 11, 679, 0, 9, 1324, 198, 330, 25, 894, 0, 45, 864, 57, 38, 127, 25, 539, 30, 257, 11, 0, 31, 0, 845, 0, 0, 0, 0, 1332, 23, 4, 17, 150, 161, 0, 15, 0, 0, 11, 7, 271, 153, 0, 25, 47, 111, 0, 160, 0, 12, 68, 69, 4, 191, 570, 99, 26, 158, 4, 88, 4, 0, 1865, 4, 23, 82, 0, 42, 0, 370, 0, 76, 194, 599, 0, 232, 0, 1850, 470, 9, 27, 0, 0, 42, 112, 25, 3987, 125, 10, 50, 5, 71, 5, 12, 235, 5, 47, 0, 84, 6, 25, 0, 156, 258, 427, 148, 0, 5, 0, 0, 214, 136, 126, 12, 55, 958, 15, 94, 15, 23, 205, 201, 0, 4, 28, 0, 11, 13, 154, 5, 24, 49, 218, 0, 17, 361, 7, 37, 66, 164, 395, 60, 41, 15, 23, 31, 1503, 187, 101, 0, 24, 240, 0, 29, 63, 15, 75, 82, 0, 30, 0, 145, 21, 221, 13, 0, 0, 4, 11, 125, 17, 68, 6, 0, 122, 29, 1052, 4, 0, 28, 17, 17, 5, 63, 0, 16, 32, 520, 0, 957, 0, 11, 25, 61, 0, 0, 1721, 92, 44, 65, 237, 563, 515, 358, 600, 124, 5, 188, 0, 228, 946, 5, 785, 21, 284, 0, 289, 74, 0, 25, 15, 19, 90, 39, 382, 7, 4694, 52, 31, 21, 429, 16, 22, 8, 0, 0, 754, 69, 190, 25, 2773, 14, 11, 30, 0, 21, 0, 15, 34, 176, 4, 23, 9, 46, 43, 145, 1013, 16, 45, 259, 9, 6, 4, 0, 0, 70, 54, 9, 0, 2461, 54, 0, 4, 4460, 4, 0, 292, 15, 37, 454, 5, 39, 89, 0, 6, 652, 42, 25, 2182, 127, 5, 399, 161, 110, 224, 194, 42, 0, 12, 0, 41, 1135, 823, 135, 0, 7, 0, 31, 11, 9, 2230, 578, 6, 0, 226, 62, 257, 0, 117, 0, 0, 16, 286, 5, 0, 19, 14, 0, 511, 150, 21, 63, 68, 0, 43, 12, 36, 377, 654, 12, 339, 66, 59, 0, 23, 11, 570, 0, 0, 94, 31, 243, 3126, 32, 195, 218, 3471, 8, 0, 1428, 270, 115, 0, 2317, 110, 39, 206, 687, 13, 1476, 97, 0, 28, 25, 2861, 221, 112, 2034, 295, 16, 980, 847, 0, 16, 191, 573, 501, 744, 724, 409, 131, 19, 932, 20, 215, 134, 47, 1310, 49, 0, 4, 123, 42, 56, 16, 39, 225, 13, 4, 118, 329, 22, 1143, 7, 102, 35, 151, 77, 624, 5, 28, 0, 23, 331, 0, 804, 33, 85, 6, 2049, 713, 20, 1190, 43, 50, 12, 0, 10, 24, 0, 181, 632, 383, 404, 14, 0, 14, 130, 0, 15, 270, 36, 11, 5, 0, 749, 4079, 23, 131, 0, 185, 17, 38, 17, 240, 314, 111, 2490, 6, 15, 152, 408, 8, 2139, 169, 11, 11, 118, 28, 0, 121, 386, 568, 27, 4, 8, 347, 8, 4733, 30, 7544, 24, 200, 56, 31, 362, 76, 1048, 23, 379, 41, 10, 38, 0, 0, 14, 27, 26, 176, 7210, 888, 0, 66, 164, 7, 0, 39, 1791, 2322, 0, 17, 313, 21, 3991, 33, 26, 38, 6, 548, 0, 727, 1559, 30, 54, 856, 0, 237, 58, 40, 10, 60, 75, 48, 104, 0, 105, 592, 1593, 32, 63, 23, 583, 0, 60, 876, 376, 23, 174, 51, 24, 16, 140, 8, 8, 108, 9, 53, 91, 409, 1901, 6, 6, 159, 493, 93, 184, 149, 858, 69, 34, 706, 324, 377, 0, 122, 0, 21, 0, 0, 11, 0, 101, 68, 18, 410, 422, 176, 0, 89, 568, 61, 262, 751, 253, 112, 0, 851, 13, 61, 4, 1227, 20, 68, 988, 11, 0, 273, 0, 96, 0, 693, 341, 95, 214, 1013, 2039, 0, 4]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Rating vs # Votes vs Cost"}, "xaxis": {"linecolor": "black", "linewidth": 2, "mirror": true, "showline": true, "title": {"text": "Rating"}}, "yaxis": {"title": {"text": "# Votes"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('7bf1f7b2-f05a-47b1-bd3f-8d53225820ec');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


**Key Takeaway:** Highly rated restaurant's are more costly and voted by users than less rated restaurant's

##### Most Frequent Words - In Reviews Text

Most Frequent Words in Online Order (Yes/No)


```python
train1_df = restaurant_df[restaurant_df["online_order"]=="Yes"].fillna("")
train0_df = restaurant_df[restaurant_df["online_order"]=="No"].fillna("")

## Get the bar chart from 1st Class ##
freq_dict = defaultdict(int)
for sent in train0_df["extracted_reviews"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

## Get the bar chart from 2nd Class ##
freq_dict = defaultdict(int)
for sent in train1_df["extracted_reviews"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of Online Order (Yes)", 
                                          "Frequent words of Online Order (No)"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')
```


<div>


            <div id="51f9e5b9-2067-4a8b-999d-426951e48c63" class="plotly-graph-div" style="height:1200px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("51f9e5b9-2067-4a8b-999d-426951e48c63")) {
                    Plotly.newPlot(
                        '51f9e5b9-2067-4a8b-999d-426951e48c63',
                        [{"marker": {"color": "blue"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [783, 789, 790, 796, 799, 843, 843, 846, 850, 859, 859, 868, 875, 918, 935, 942, 961, 975, 987, 990, 1044, 1087, 1099, 1116, 1138, 1152, 1153, 1161, 1214, 1280, 1305, 1587, 1661, 1761, 1865, 2006, 2180, 2260, 2799, 2833, 2862, 2952, 2954, 2992, 3160, 3302, 8326, 9205, 10838, 11782], "xaxis": "x", "y": ["crowd", "perfect", "tea", "bangalore", "little", "worth", "cake", "definitely", "chocolate", "small", "bad", "quality", "tasty", "awesome", "look", "dish", "pretty", "decent", "friendly", "nthe", "veg", "come", "menu", "music", "coffee", "amazing", "friend", "drink", "pizza", "experience", "price", "restaurant", "serve", "best", "staff", "love", "time", "nice", "visit", "ambience", "chicken", "taste", "great", "service", "try", "order", "food", "good", "place", "ratedn"], "yaxis": "y"}, {"marker": {"color": "blue"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [1400, 1404, 1420, 1431, 1450, 1453, 1467, 1470, 1479, 1482, 1536, 1553, 1562, 1618, 1657, 1688, 1725, 1803, 1825, 1864, 1920, 1966, 1975, 1996, 2075, 2212, 2313, 2319, 2396, 2449, 2721, 2878, 2981, 3339, 3382, 3432, 3649, 4090, 4384, 4410, 4433, 5327, 6613, 6652, 6861, 8159, 15634, 16312, 17916, 23116], "xaxis": "x2", "y": ["small", "lunch", "friend", "definitely", "worth", "paneer", "cheese", "pretty", "nthe", "decent", "awesome", "fry", "recommend", "rice", "fish", "delicious", "look", "friendly", "tasty", "quantity", "bad", "amazing", "price", "quality", "menu", "experience", "pizza", "dish", "come", "biryani", "veg", "best", "staff", "serve", "restaurant", "love", "nice", "time", "visit", "great", "ambience", "service", "chicken", "try", "taste", "order", "food", "place", "good", "ratedn"], "yaxis": "y2"}],
                        {"annotations": [{"font": {"size": 16}, "showarrow": false, "text": "Frequent words of Online Order (Yes)", "x": 0.225, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}, {"font": {"size": 16}, "showarrow": false, "text": "Frequent words of Online Order (No)", "x": 0.775, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}], "height": 1200, "paper_bgcolor": "rgb(233,233,233)", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Word Count Plots"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 0.45]}, "xaxis2": {"anchor": "y2", "domain": [0.55, 1.0]}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0]}, "yaxis2": {"anchor": "x2", "domain": [0.0, 1.0]}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('51f9e5b9-2067-4a8b-999d-426951e48c63');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


Most Frequent Words in Book Table (Yes/No)


```python
train1_df = restaurant_df[restaurant_df["book_table"]=="Yes"].fillna("")
train0_df = restaurant_df[restaurant_df["book_table"]=="No"].fillna("")

## Get the bar chart from 1st Class ##
freq_dict = defaultdict(int)
for sent in train0_df["extracted_reviews"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'red')

## Get the bar chart from 2nd Class ##
freq_dict = defaultdict(int)
for sent in train1_df["extracted_reviews"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'red')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of Book Table (Yes)", 
                                          "Frequent words of Book Table (No)"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')
```


<div>


            <div id="86efd67b-3a68-476d-8609-5e7d4df43116" class="plotly-graph-div" style="height:1200px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("86efd67b-3a68-476d-8609-5e7d4df43116")) {
                    Plotly.newPlot(
                        '86efd67b-3a68-476d-8609-5e7d4df43116',
                        [{"marker": {"color": "red"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [1297, 1304, 1324, 1391, 1392, 1398, 1401, 1432, 1446, 1461, 1472, 1519, 1528, 1538, 1546, 1555, 1573, 1580, 1617, 1652, 1719, 1741, 1799, 1825, 1880, 1986, 1997, 2041, 2259, 2276, 2374, 2882, 2882, 2907, 3015, 3163, 3175, 3364, 3583, 3697, 4134, 4351, 5970, 6136, 7030, 7971, 14503, 14641, 15953, 23753], "xaxis": "x", "y": ["little", "recommend", "meal", "friend", "eat", "money", "paneer", "pizza", "coffee", "worth", "fry", "menu", "delicious", "amazing", "delivery", "look", "chocolate", "awesome", "small", "friendly", "rice", "come", "dish", "experience", "tasty", "quantity", "veg", "staff", "bad", "quality", "price", "best", "biryani", "serve", "ambience", "nice", "love", "restaurant", "great", "visit", "time", "service", "chicken", "try", "taste", "order", "food", "place", "good", "ratedn"], "yaxis": "y"}, {"marker": {"color": "red"}, "orientation": "h", "showlegend": false, "type": "bar", "x": [945, 949, 969, 991, 992, 995, 997, 1009, 1023, 1042, 1067, 1074, 1090, 1103, 1105, 1138, 1166, 1182, 1250, 1298, 1325, 1462, 1520, 1558, 1580, 1605, 1655, 1667, 1742, 1756, 1757, 1768, 2093, 2095, 2136, 2263, 2746, 2783, 2805, 3486, 3490, 3505, 3676, 3781, 3968, 4251, 9457, 11145, 11168, 12509], "xaxis": "x2", "y": ["dessert", "bangalore", "recommend", "overall", "perfect", "definitely", "nservice", "pasta", "ambiance", "main", "buffet", "cheese", "average", "table", "look", "friendly", "decent", "friend", "starter", "nthe", "pretty", "dish", "drink", "beer", "amazing", "restaurant", "menu", "experience", "come", "music", "best", "veg", "serve", "pizza", "time", "love", "nice", "taste", "staff", "visit", "order", "chicken", "try", "great", "service", "ambience", "food", "ratedn", "good", "place"], "yaxis": "y2"}],
                        {"annotations": [{"font": {"size": 16}, "showarrow": false, "text": "Frequent words of Book Table (Yes)", "x": 0.225, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}, {"font": {"size": 16}, "showarrow": false, "text": "Frequent words of Book Table (No)", "x": 0.775, "xanchor": "center", "xref": "paper", "y": 1.0, "yanchor": "bottom", "yref": "paper"}], "height": 1200, "paper_bgcolor": "rgb(233,233,233)", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Word Count Plots"}, "width": 900, "xaxis": {"anchor": "y", "domain": [0.0, 0.45]}, "xaxis2": {"anchor": "y2", "domain": [0.55, 1.0]}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0]}, "yaxis2": {"anchor": "x2", "domain": [0.0, 1.0]}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('86efd67b-3a68-476d-8609-5e7d4df43116');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


##### WordCloud of Restaurant's Reviews


```python
plot_wordcloud(restaurant_df["extracted_reviews"], title="Word Cloud of Restaurant's Reviews")
```


![png](output_80_0.png)


**Key Takeaway:** `good`, `food` etc. are mostly written words in restaurant's reviews

### Recommendation Engine

**Input Features:**

- Reviews Text
- Location
- Budget
- Cuisine Type
    
**Output:**
- Restaurant 1 (90% likely) 
- Restaurant 2 (84% likely) 
- Restaurant 3 (81.5% likely) 

#### Data Preparation for Recommendation Engine



```python
"""
Selecting Relevant Features:
"""
input_df = restaurant_df[["restaurant_id", "restaurant_name", "extracted_reviews", "location", "cost_for_two_people", "cuisines", "rating", "no_of_votes",
                          "extracted_ratings"]]

## Defining Text/Categorical/Numerical features
cat_txt_feats = ["restaurant_id", "extracted_reviews", "location", "cuisines"]
num_feats = ["cost_for_two_people", "rating", "extracted_ratings", "no_of_votes"]

## Categorical & Text Features: Fill NULL's with Blank
input_df[cat_txt_feats] = input_df[cat_txt_feats].fillna("")

## Fill Cost NULL values with Mean
input_df["cost_for_two_people"] = input_df["cost_for_two_people"].apply(lambda x: float(str(x).replace(',','')))
input_df["cost_for_two_people"] = input_df["cost_for_two_people"].fillna(np.mean(input_df["cost_for_two_people"]))

"""
Filling rating (NULL/BLANK) values with extracted_ratings and filling remaining NULL's with -1
"""
input_df["rating"] = np.where(input_df["rating"].isnull(), input_df["extracted_ratings"], input_df["rating"])
input_df["rating"] = np.round(input_df["rating"].fillna(-1),1)

## Drop extracted_ratings columns:
input_df.drop(["extracted_ratings"], axis = 1, inplace = True)
```

#### User Input Parameters


```python
"""
Define User's Inputs Here
"""
USER_LOCATION = "Jayanagar"
USER_CUISINES = "North Indian"
USER_BUDGET = 800
USER_DESCRIPTION = "good ambiance restaurants, serving fish"
USER_DESCRIPTION = gensim_clean_text(USER_DESCRIPTION) ## Clean User Description
```

####  Content Based Recommender System

Computes the similarity between restaurants based on certain parameter/metric(s) and suggests restaurants that are most similar to a particular restaurant that a user liked (user input).

As a first step, to reduce the search space, we will take user inputs (Location, Cuisines, Budget) and filter the dataset. After applying these filteres we will check the cosine similarity between User Description and Restaurant Review (Using TF-IDF vectorizer). Then, we will come up with top 3 restaurant's based on User's Input Parameter's

*Other things can be tried: Semantic based similarity between User's description and Reviews*


```python
gradient_dataframe(restaurant_recommend_func(input_df), "Top 3 Restaurant's based on User's 4 Input Parameters")
```

    Dataset Shape after applying Location, Cuisines and Budget filters:  (18, 8)
    




<style  type="text/css" >
    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56 tr:nth-of-type(odd) {
          background: #eee;
          text-align: left;
    }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56 tr:nth-of-type(even) {
          background: white;
          text-align: left;
    }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56 th {
          background: #808080;
          color: white;
          font-family: verdana;
          text-align: left;
    }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56 td {
          font-family: verdana;
          text-align: left;
    }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row0_col0 {
            background-color:  #08306b;
            color:  #f1f1f1;
        }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row0_col4 {
            background-color:  #08306b;
            color:  #f1f1f1;
        }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row0_col5 {
            background-color:  #08306b;
            color:  #f1f1f1;
        }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row0_col6 {
            background-color:  #08306b;
            color:  #f1f1f1;
        }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row1_col0 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row1_col4 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row1_col5 {
            background-color:  #08306b;
            color:  #f1f1f1;
        }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row1_col6 {
            background-color:  #08306b;
            color:  #f1f1f1;
        }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row2_col0 {
            background-color:  #9ac8e0;
            color:  #000000;
        }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row2_col4 {
            background-color:  #08306b;
            color:  #f1f1f1;
        }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row2_col5 {
            background-color:  #f7fbff;
            color:  #000000;
        }    #T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row2_col6 {
            background-color:  #f7fbff;
            color:  #000000;
        }</style><table id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56" ><caption>Top 3 Restaurant's based on User's 4 Input Parameters</caption><thead>    <tr>        <th class="col_heading level0 col0" >Restaurant ID</th>        <th class="col_heading level0 col1" >Restaurant Name</th>        <th class="col_heading level0 col2" >Location</th>        <th class="col_heading level0 col3" >Cuisines</th>        <th class="col_heading level0 col4" >Cost</th>        <th class="col_heading level0 col5" >Rating</th>        <th class="col_heading level0 col6" >% Likely</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row0_col0" class="data row0 col0" >21882</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row0_col1" class="data row0 col1" >Shanmukha</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row0_col2" class="data row0 col2" >Jayanagar</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row0_col3" class="data row0 col3" >Biryani, Andhra, North Indian, Chinese</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row0_col4" class="data row0 col4" >800.000000</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row0_col5" class="data row0 col5" >4.400000</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row0_col6" class="data row0 col6" >7.694331</td>
            </tr>
            <tr>
                                <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row1_col0" class="data row1 col0" >2547</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row1_col1" class="data row1 col1" >Empire Restaurant</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row1_col2" class="data row1 col2" >Jayanagar</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row1_col3" class="data row1 col3" >North Indian, Mughlai, South Indian, Chinese</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row1_col4" class="data row1 col4" >750.000000</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row1_col5" class="data row1 col5" >4.400000</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row1_col6" class="data row1 col6" >7.689718</td>
            </tr>
            <tr>
                                <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row2_col0" class="data row2 col0" >9985</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row2_col1" class="data row2 col1" >Hyderabad Bawarchi</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row2_col2" class="data row2 col2" >Jayanagar</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row2_col3" class="data row2 col3" >North Indian, Biryani, Chinese</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row2_col4" class="data row2 col4" >800.000000</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row2_col5" class="data row2 col5" >3.900000</td>
                        <td id="T_536a5fe8_e9c2_11ea_b481_7470fd5d5a56row2_col6" class="data row2 col6" >6.035753</td>
            </tr>
    </tbody></table>



**Key Takeaway:** `Shanmukha`, `Empire Restaurant`, `Hyderabad Bawarchi` are top 3 recommended restaurant by our algorithm based on user's input paramerters
