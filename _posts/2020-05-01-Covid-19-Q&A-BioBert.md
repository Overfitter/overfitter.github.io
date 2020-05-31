<font size="5" >🔍Know-Corona : COVID-19 Open Research Dataset Challenge </font>

### Loading metadata dataframe


```python
!pip install rank_bm25 -q

import numpy as np
import pandas as pd 
from pathlib import Path, PurePath

import nltk
from nltk.corpus import stopwords
import re
import string
import torch

from rank_bm25 import BM25Okapi
```


```python
"""
Load metadata
"""

data_dir = PurePath('../data')
metadata_path = data_dir / 'covid_19_metadata.csv'
metadata_df = pd.read_csv(metadata_path,
                               dtype={'Microsoft Academic Paper ID': str, 'pubmed_id': str})
metadata_df = metadata_df.dropna(subset=['abstract', 'title']).reset_index(drop=True)
```

### 3. Covid Search Engine


```python
from rank_bm25 import BM25Okapi

# adapted from https://www.kaggle.com/dgunning/building-a-cord19-research-engine-with-bm25
english_stopwords = list(set(stopwords.words('english')))

class CovidSearchEngine:
    """
    Simple CovidSearchEngine.
    
    Usage:
    
    cse = CovidSearchEngine(metadata_df) # metadata_df is a pandas dataframe with 'title' and 'abstract' columns 
    search_results = cse.search("What is coronavirus", num=10) # Return `num` top-results
    """
    
    def remove_special_character(self, text):
        """
        Remove all special character from text string
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text):
        """
        Tokenize with NLTK

        Rules:
            - drop all words of 1 and 2 characters
            - drop all stopwords
            - drop all numbers
        """
        words = nltk.word_tokenize(text)
        return list(set([word for word in words 
                         if len(word) > 1
                         and not word in english_stopwords
                         and not word.isnumeric() 
                        ])
                   )
    
    def preprocess(self, text):
        """
        Clean and tokenize text input
        """
        return self.tokenize(self.remove_special_character(text.lower()))


    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus
        self.columns = corpus.columns
        raw_search_str = self.corpus.abstract.fillna('') + ' ' + self.corpus.title.fillna('')
        self.index = raw_search_str.apply(self.preprocess).to_frame()
        self.index.columns = ['terms']
        self.index.index = self.corpus.index
        self.bm25 = BM25Okapi(self.index.terms.tolist())
    
    def search(self, query, num):
        """
        Return top `num` results that better match the query
        """
        search_terms = self.preprocess(query) 
        doc_scores = self.bm25.get_scores(search_terms) # get scores
        
        ind = np.argsort(doc_scores)[::-1][:num] # sort results
        
        results = self.corpus.iloc[ind][self.columns] # Initialize results_df
        results['score'] = doc_scores[ind] # Insert 'score' column
        results = results[results.score > 0]
        return results.reset_index()
    
cse = CovidSearchEngine(metadata_df) # Covid Search Engine
```

### Question-Answering system (BioBert)


```python
%%time

"""
LIBRARIES
"""

import torch
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

"""
SETTINGS
"""

NUM_CONTEXT_FOR_EACH_QUESTION = 10


"""
Transformers
"""

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Code running on: {}".format(torch_device) )

model = AutoModelForQuestionAnswering.from_pretrained('ktrapeznikov/biobert_v1.1_pubmed_squad_v2')
tokenizer = AutoTokenizer.from_pretrained('ktrapeznikov/biobert_v1.1_pubmed_squad_v2')

model = model.to(torch_device)
model.eval()

def answer_question(question, context):
    """
    Answer questions
    """
    encoded_dict = tokenizer.encode_plus(
                        question, context, # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 256,  # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt'     # Return pytorch tensors.
                   )
    
    input_ids = encoded_dict['input_ids'].to(torch_device)
    token_type_ids = encoded_dict['token_type_ids'].to(torch_device) # segments
    
    start_scores, end_scores = model(input_ids, token_type_ids=token_type_ids)

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = tokenizer.convert_tokens_to_string(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
    
    answer = answer.replace('[CLS]', '')
    
    return answer



from transformers import BartTokenizer, BartModel

tokenizer_summarize = BartTokenizer.from_pretrained('bart-large')
model_summarize = BartModel.from_pretrained('bart-large').to(torch_device)


model_summarize.to(torch_device)
# Set the model in evaluation mode to deactivate the DropOut modules
model_summarize.eval()

def get_summary(text):
    """
    Get summary
    """
    
    answers_input_ids = tokenizer_summarize.batch_encode_plus(
        [text], return_tensors='pt', max_length=1024
    )['input_ids']
    
    answers_input_ids = answers_input_ids.to(torch_device)
    
    summary_ids = model_summarize.generate(answers_input_ids,
                                           num_beams=4,
                                           max_length=5,
                                           early_stopping=True
                                          )
        
    return tokenizer_summarize.decode(summary_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=False)

    
"""
Main 
"""



def create_output_results(question, all_contexts, all_answers, summary_answer, summary_context):
    """
    Return a dictionary of the form
    
    {
        question: 'what is coronavirus',
        results: [
            {
                'context': 'coronavirus is an infectious disease caused by',
                'answer': 'infectious disease'
                'start_index': 18
                'end_index': 36
            },
            {
                ...
            }
        ]
    }
    
    Start and end index are useful to find the position of the answer in the context  
    """
    
    def find_start_end_index_substring(context, answer):   
        search_re = re.search(re.escape(answer.lower()), context.lower())
        if search_re:
            return search_re.start(), search_re.end()
        else:
            return 0, len(context)
        
    output = {}
    output['question'] = question
    output['summary_answer'] = summary_answer
    output['summary_context'] = summary_context
    results = []
    for c, a in zip(all_contexts, all_answers):

        span = {}
        span['context'] = c
        span['answer'] = a
        span['start_index'], span['end_index'] = find_start_end_index_substring(c,a)

        results.append(span)
    
    output['results'] = results
        
    return output


def get_all_context(query, num_results):
    """
    Search in the metadata dataframe and return the first `num` results that better match the query 
    """
    
    papers_df = cse.search(query, num_results)
    return papers_df['abstract'].str.replace("Abstract", "").tolist()


def get_all_answers(question, all_context):
    """
    Return a list of all answers, given a question and a list of context
    """    
    
    all_answers = []
    
    for context in all_context:
        all_answers.append(answer_question(question, context))
    return all_answers

    
def get_results(question, summarize=False, num_results=NUM_CONTEXT_FOR_EACH_QUESTION, verbose=True):
    """
    Return dict object containg a list of all context and answers related to the (sub)question
    """
    
    if verbose:
        print("Getting context ...")
    all_contexts = get_all_context(question, num_results)
    
    if verbose:
        print("Answering to all questions ...")
    all_answers = get_all_answers(question, all_contexts)
    
    summary_answer = ''
    summary_context = ''
    if verbose and summarize:
        print("Adding summary ...")
    if summarize:
        summary_answer = get_summary(all_answers)
        summary_context = get_summary(all_contexts)
    
    if verbose:
        print("output.")
    
    return create_output_results(question, all_contexts, all_answers, summary_answer, summary_context)
```

    Code running on: cpu
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1264.0, style=ProgressStyle(description…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=1625270110.0, style=ProgressStyle(descr…


    
    Wall time: 7min 28s
    

### Dict object to store all Kaggle CORD-19 tasks


```python
# adapted from https://www.kaggle.com/dirktheeng/anserini-bert-squad-for-semantic-corpus-search

covid_kaggle_questions = {
"data":[
          {
              "task": "What is known about transmission, incubation, and environmental stability?",
              "questions": [
                  "Is the virus transmitted by aerisol, droplets, food, close contact, fecal matter, or water?",
                  "How long is the incubation period for the virus?",
                  "Can the virus be transmitted asymptomatically or during the incubation period?",
                  "How does weather, heat, and humidity affect the tramsmission of 2019-nCoV?",
                  "How long can the 2019-nCoV virus remain viable on common surfaces?"
              ]
          },
          {
              "task": "What do we know about COVID-19 risk factors?",
              "questions": [
                  "What risk factors contribute to the severity of 2019-nCoV?",
                  "How does hypertension affect patients?",
                  "How does heart disease affect patients?",
                  "How does copd affect patients?",
                  "How does smoking affect patients?",
                  "How does pregnancy affect patients?",
                  "What is the fatality rate of 2019-nCoV?",
                  "What public health policies prevent or control the spread of 2019-nCoV?"
              ]
          },
          {
              "task": "What do we know about virus genetics, origin, and evolution?",
              "questions": [
                  "Can animals transmit 2019-nCoV?",
                  "What animal did 2019-nCoV come from?",
                  "What real-time genomic tracking tools exist?",
                  "What geographic variations are there in the genome of 2019-nCoV?",
                  "What effors are being done in asia to prevent further outbreaks?"
              ]
          },
          {
              "task": "What do we know about vaccines and therapeutics?",
              "questions": [
                  "What drugs or therapies are being investigated?",
                  "Are anti-inflammatory drugs recommended?"
              ]
          },
          {
              "task": "What do we know about non-pharmaceutical interventions?",
              "questions": [
                  "Which non-pharmaceutical interventions limit tramsission?",
                  "What are most important barriers to compliance?"
              ]
          },
          {
              "task": "What has been published about medical care?",
              "questions": [
                  "How does extracorporeal membrane oxygenation affect 2019-nCoV patients?",
                  "What telemedicine and cybercare methods are most effective?",
                  "How is artificial intelligence being used in real time health delivery?",
                  "What adjunctive or supportive methods can help patients?"
              ]
          },
          {
              "task": "What do we know about diagnostics and surveillance?",
              "questions": [
                  "What diagnostic tests (tools) exist or are being developed to detect 2019-nCoV?"
              ]
          },
          {
              "task": "Other interesting questions",
              "questions": [
                  "What is the immune system response to 2019-nCoV?",
                  "Can personal protective equipment prevent the transmission of 2019-nCoV?",
                  "Can 2019-nCoV infect patients a second time?"
              ]
          }
   ]
}
```

### Answer to all questions

Store it in the `all_answers` dataframe.


```python
all_tasks = []


for i, t in enumerate(covid_kaggle_questions['data']):
    print("Answering question to task {}. ...".format(i+1))
    answers_to_question = []
    for q in t['questions']:
            answers_to_question.append(get_results(q, verbose=False))
    task = {}
    task['task'] = t['task']
    task['questions'] = answers_to_question
    
    all_tasks.append(task)

all_answers = {}
all_answers['data'] = all_tasks

```

    Answering question to task 1. ...
    Answering question to task 2. ...
    Answering question to task 3. ...
    Answering question to task 4. ...
    Answering question to task 5. ...
    Answering question to task 6. ...
    Answering question to task 7. ...
    Answering question to task 8. ...
    

###  Display questions, context and answers


```python
# Adapted from https://jbesomi.github.io/Korono/

from IPython.display import display, Markdown, Latex, HTML

def layout_style():
    
    
    style = """
        div {
            color: black;
        }
        
        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }
        
        .answer{
            color: #dc7b15;
        }
        
        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }
               
        div.output_scroll { 
            height: auto; 
        }
    
    """
    
    return "<style>" + style + "</style>"

def dm(x): display(Markdown(x))
def dh(x): display(HTML(layout_style() + x))
    
def display_task(task):
    m("## " + task['task'])
    
#display_task(task1['data'][0])


def display_single_context(context, start_index, end_index):
    
    before_answer = context[:start_index]
    answer = context[start_index:end_index]
    after_answer = context[end_index:]

    content = before_answer + "<span class='answer'>" + answer + "</span>" + after_answer

    return dh("""<div class="single_answer">{}</div>""".format(content))

def display_question_title(question):
    return dh("<h2 class='question_title'>{}</h2>".format(question.capitalize()))

def answer_not_found(context, start_index, end_index):
    return (start_index == 0 and len(context) == end_index) or (start_index == 0 and end_index == 0)
def display_all_context(index, question):
    
    display_question_title(str(index + 1) + ". " + question['question'].capitalize())
    
    # display context
    for i in question['results']:
        if answer_not_found(i['context'], i['start_index'], i['end_index']):
            continue # skip not found questions
        display_single_context(i['context'], i['start_index'], i['end_index'])

def display_task_title(index, task):
    task_title = "Task " + str(index) + ": " + task
    return dh("<h1 class='task_title'>{}</h1>".format(task_title))

def display_single_task(index, task):
    
    display_task_title(index, task['task'])
    
    for i, question in enumerate(task['questions']):
        display_all_context(i, question)

task = 1
display_single_task(task, all_tasks[task-1])
```


```python
task = 2
display_single_task(task, all_tasks[task-1])
```


<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h1 class='task_title'>Task 2: What do we know about COVID-19 risk factors?</h1>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>1. what risk factors contribute to the severity of 2019-ncov?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer"><span class='answer'>Diabetes mellitus and hypertension</span> are recognized risk factors for severe clinical outcomes including death associated with Middle East respiratory syndrome coronavirus infection Among 32 virusinfected patients in Saudi Arabia severity of illness and frequency of death corresponded closely with presence of multiple and more severe underlying conditions</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>2. how does hypertension affect patients?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>3. how does heart disease affect patients?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">BACKGROUND Studies have reminded that cardiovascular metabolic comorbidities made patients more susceptible to suffer 2019 novel corona virus 2019nCoV disease COVID19 and <span class='answer'>exacerbated the infection</span> The aim of this analysis is to determine the association of cardiovascular metabolic diseases with the development of COVID19 METHODS A metaanalysis of eligible studies that summarized the prevalence of cardiovascular metabolic diseases in COVID19 and compared the incidences of the comorbidities in ICUsevere and nonICUsevere patients was performed Embase and PubMed were searched for relevant studies RESULTS A total of six studies with 1527 patients were included in this analysis The proportions of hypertension cardiacerebrovascular disease and diabetes in patients with COVID19 were 171 164 and 97 respectively The incidences of hypertension cardiacerebrovascular diseases and diabetes were about twofolds threefolds and twofolds respectively higher in ICUsevere cases than in their nonICUsevere counterparts At least 80 patients with COVID19 suffered the acute cardiac injury The incidence of acute cardiac injury was about 13 folds higher in ICUsevere patients compared with the nonICUsevere patients CONCLUSION Patients with previous cardiovascular metabolic diseases may face a greater risk of developing into the severe condition and the comorbidities can also greatly affect the prognosis of the COVID19 On the other hand COVID19 can in turn aggravate the damage to the heart</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>4. how does copd affect patients?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>5. how does smoking affect patients?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>6. how does pregnancy affect patients?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">INTRODUCTION Noninvasive ventilation NIV is not proven to be effective in treating respiratory failure in severe pneumonia However some clinicians nevertheless attempt NIV to indirectly deliver adequate oxygenation and avoid unnecessary endotracheal intubation CASE PRESENTATION In this article we report the case of a 24yearold woman at 32 weeks gestation who presented with hypoxemic respiratory failure requiring mechanical ventilation She was successfully managed by NIV DISCUSSION However NIV must be managed by providers who are trained in mechanical ventilation This is of the utmost importance in avoiding any delay should the patients condition worsen and require endotracheal intubation Moreover in pregnant women the <span class='answer'>severity of illness may progress quickly due to the immunosuppression</span> inherent in these patients CONCLUSION Special attention should be given to the choices of invasive ventilation and NIV to manage community acquired pneumonia patients in third trimester</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>7. what is the fatality rate of 2019-ncov?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Since end of December 2019 a cluster of patients with pneumonia of unknown origin was reported from Wuhan Hubei province China They shared a connection with the Huanan South China Seafood Market in Wuhan and now it has been confirmed that the disease is caused by a novel coronavirus provisionally named 2019nCoV As of today 30 January 2020 7734 cases have been confirmed in China and 90 cases have also been cumulatively reported from Taiwan Thailand Vietnam Malaysia Nepal Sri Lanka Cambodia Japan Singapore Republic of Korea United Arab Emirate United States The Philippines India Australia Canada Finland France and Germany Finland France and Germany are the only European countries in which cases [n 1 n  5 and n  4 respectively] have been reported up to date According to the released news the case rate fatality is <span class='answer'>22 1707824</span></div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Objective We aim to summarize reliable evidences of evidencebased medicine for the treatment and prevention of the 2019 novel coronavirus 2019nCoV by analyzing all the published studies on the clinical characteristics of patients with 2019nCoV Methods PubMed Cochrane Library Embase and other databases were searched Several studies on the clinical characteristics of 2019nCoV infection were collected for Metaanalysis Results Ten studies were included in Metaanalysis including a total number of 50466 patients with 2019nCoV infection Metaanalysis shows that among these patients the incidence of fever was 891 the incidence of cough was 722 and the incidence of muscle soreness or fatigue was 425 The incidence of acute respiratory distress syndrome ARDS was 148 the incidence of abnormal chest computer tomography CT was 966 the percentage of severe cases in all infected cases was 181 and the case fatality rate of patients with 2019nCoV infection was <span class='answer'>43</span> Conclusion Fever and cough are the most common symptoms in patients with 2019nCoV infection and most of these patients have abnormal chest CT examination Several people have muscle soreness or fatigue as well as ARDS Diarrhea hemoptysis headache sore throat shock and other symptoms only occur in a small number of patients The case fatality rate of patients with 2019nCoV infection is lower than that of Severe Acute Respiratory Syndrome SARS and Middle East Respiratory Syndrome MERS</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer"> Objective We aim to summarize reliable evidences of evidencebased medicine for the treatment and prevention of the 2019 novel coronavirus 2019nCoV by analyzing all the published studies on the clinical characteristics of patients with 2019nCoV Methods PubMed Cochrane Library Embase and other databases were searched Several studies on the clinical characteristics of 2019nCoV infection were collected for Metaanalysis Results Ten studies were included in Metaanalysis including a total number of 50466 patients with 2019nCoV infection Metaanalysis shows that among these patients the incidence of fever was 891 the incidence of cough was 722 and the incidence of muscle soreness or fatigue was 425 The incidence of acute respiratory distress syndrome ARDS was 148 the incidence of abnormal chest computer tomography CT was 966 the percentage of severe cases in all infected cases was 181 and the case fatality rate of patients with 2019nCoV infection was <span class='answer'>43</span> Conclusion Fever and cough are the most common symptoms in patients with 2019nCoV infection and most of these patients have abnormal chest CT examination Several people have muscle soreness or fatigue as well as ARDS Diarrhea hemoptysis headache sore throat shock and other symptoms only occur in a small number of patients The case fatality rate of patients with 2019nCoV infection is lower than that of Severe Acute Respiratory Syndrome SARS and Middle East Respiratory Syndrome MERS This article is protected by copyright All rights reserved</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">We present a timely evaluation of the Chinese 2019nCov epidemic in its initial phase where 2019nCov demonstrates comparable transmissibility but <span class='answer'>lower fatality rates</span> than SARS and MERS A quick diagnosis that leads to case isolation and integrated interventions will have a major impact on its future trend Nevertheless as China is facing its Spring Festival travel rush and the epidemic has spread beyond its borders further investigation on its potential spatiotemporal transmission pattern and novel intervention strategies are warranted</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Through literature review and group discussion Special Expert Group for Control of the Epidemic of Novel Coronavirus Pneumonia of the Chinese Preventive Medicine Association formulated an update on the epidemiological characteristics of novel coronavirus pneumonia NCP The initial source of the 2019 novel coronavirus 2019nCoV was the Huanan seafood market in Wuhan Hubei province China with pangolins as a potential animal host Currently the main source of infection is NCP patients and asymptomatic carriers may also be infectious The virus is believed transmitted mostly via droplets or contact People are all generally susceptible to the virus The average incubation period was 52 days and the basic reproductive number R0 was 22 at the onset of the outbreak Most NCP patients were clinically mild cases The case fatality rate was <span class='answer'>238</span> and elderly men with underlying diseases were at a higher risk of death Strategies for prevention and control of NCP include improving epidemic surveillance quarantining the source of infection speeding up the diagnosis of suspected cases optimizing the management of close contacts tightening prevention and control of cluster outbreaks and hospital infection preventing possible rebound of the epidemic after people return to work from the Chinese Spring Festival holiday and strengthening community prevention and control</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>8. what public health policies prevent or control the spread of 2019-ncov?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">On December 31 2019 the World Health Organization was notified about a cluster of pneumonia of unknown aetiology in the city of Wuhan China Chinese authorities later identified a new coronavirus 2019nCoV as the causative agent of the outbreak As of January 23 2020 655 cases have been confirmed in China and several other countries Understanding the transmission characteristics and the potential for sustained humantohuman transmission of 2019nCoV is critically important for coordinating current screening and containment strategies and determining whether the outbreak constitutes a public health emergency of international concern PHEIC We performed stochastic simulations of early outbreak trajectories that are consistent with the epidemiological findings to date We found the basic reproduction number R0 to be around 22 90 high density interval 1438 indicating the potential for sustained humantohuman transmission Transmission characteristics appear to be of a similar magnitude to severe acute respiratory syndromerelated coronavirus SARSCoV and the 1918 pandemic influenza These findings underline the importance of <span class='answer'>heightened screening surveillance and control efforts</span> particularly at airports and other travel hubs in order to prevent further international spread of 2019nCoV</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Infections with 2019nCoV can spread from person to person and in the earliest phase of the outbreak the basic reproductive number was estimated to be around 22 assuming a mean serial interval of 75 days [2] The serial interval was not precisely estimated and a potentially shorter mean serial interval would have corresponded to a slightly lower basic reproductive number <span class='answer'>Control measures</span> and changes in population behaviour later in January should have reduced the effective reproductive number However it is too early to estimate whether the effective reproductive number has been reduced to below the critical threshold of 1 because cases currently being detected and reported would have mostly been infected in mid to lateJanuary Average delays between infection and illness onset have been estimated at around 56 days with an upper limit of around 1114 days [25] and delays from illness onset to laboratory confirmation added a further 10 days on average [2]</div>



```python
task = 3
display_single_task(task, all_tasks[task-1])
```


<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h1 class='task_title'>Task 3: What do we know about virus genetics, origin, and evolution?</h1>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>1. can animals transmit 2019-ncov?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">The outbreak of pneumonia caused by the novel coronavirus 2019nCoV in Wuhan Hubei province of China at the end of 2019 shaped tremendous challenges to Chinas public health and clinical treatment The virus belongs to the  genus Coronavirus in the family Corornaviridae and is closely related to SARSCoV and MERSCoV causing severe symptoms of pneumonia The virus is transmitted through droplets close contact and other means and <span class='answer'>patients in the incubation period could potentially transmit the virus to other persons</span> According to current observations 2019nCoV is weaker than SARS in pathogenesis but has stronger transmission competence its mechanism of crossspecies spread might be related with angiotensinconverting enzyme  ACE2 which is consistent with the receptor SARSCoV After the outbreak of this disease Chinese scientists invested a lot of energy to carry out research by developing rapid diagnostic reagents identifying the characters of the pathogen screening out clinical drugs that may inhibit the virus and are rapidly developing vaccines The emergence of 2019nCoV reminds us once again of the importance of establishing a systematic coronavirus surveillance network It also poses new challenges to prevention and control of the emerging epidemic and rapidly responses on scientific research</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>2. what animal did 2019-ncov come from?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>3. what real-time genomic tracking tools exist?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Pathogen detection identification and tracking is shifting from nonmolecular methods DNA fingerprinting methods and single gene methods to methods relying on whole genomes <span class='answer'>Viral Ebola and influenza genome data</span> are being used for realtime tracking while foodborne bacterial pathogen outbreaks and hospital outbreaks are investigated using whole genomes in the UK Canada the USA and the other countries Also plant pathogen genomes are starting to be used to investigate plant disease epidemics such as the wheat blast outbreak in Bangladesh While these genomebased approaches provide neverseen advantages over all previous approaches with regard to public health and biosecurity they also come with new vulnerabilities and risks with regard to cybersecurity The more we rely on genome databases the more likely these databases will become targets for cyberattacks to interfere with public health and biosecurity systems by compromising their integrity taking them hostage or manipulating the data they contain Also while there is the potential to collect pathogen genomic data from infected individuals or agricultural and food products during disease outbreaks to improve disease modeling and forecast how to protect the privacy of individuals growers and retailers is another major cyberbiosecurity challenge As data become linkable to other data sources individuals and groups become identifiable and potential malicious activities targeting those identified become feasible Here we define a number of potential cybersecurity weaknesses in todays pathogen genome databases to raise awareness and we provide potential solutions to strengthen cyberbiosecurity during the development of the next generation of pathogen genome databases</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer"> Avian rotaviruses AvRVs represent a diverse group of intestinal viruses which are suspected as the cause of several diseases in poultry with symptoms of diarrhoea growth retardation or runting and stunting syndrome RSS To assess the distribution of AvRVs in chickens and turkeys we have developed specific PCR protocols These protocols were applied in two field studies investigating faecal samples or intestinal contents of diseased birds derived from several European countries and Bangladesh In the first study samples of 166 chickens and 33 turkeys collected between 2005 and 2008 were tested by PAGE and conventional RTPCR and AvRVs were detected in 462 In detail 161 and 392 were positive for AvRVs of groups A or D respectively 111 of the samples contained both of them and only four samples 20 contained rotaviruses showing a PAGE pattern typical for groups F and G In the second study samples from 375 chickens and 18 turkeys collected between 2009 and 2010 were analyzed using a more sensitive <span class='answer'>group Aspecific and a new group Dspecific realtime RTPCR</span> In this survey 850 were AvRVpositive 588 for group A AvRVs 659 for group D AvRVs and 389 for both of them Although geographical differences exist the results generally indicate a very high prevalence of group A and D rotaviruses in chicken and turkey flocks with cases of diarrhoea growth retardation or RSS The newly developed diagnostic tools will help to investigate the epidemiology and clinical significance of AvRV infections in poultry</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>4. what geographic variations are there in the genome of 2019-ncov?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">To investigate the genetic diversity time origin and evolutionary history of the 2019nCoV outbreak in China and Thailand a total of 12 genome sequences of the virus with known sampling date 24 December 2019 and 13 January 2020 and geographic location primarily <span class='answer'>Wuhan city Hubei Province China</span> but also Bangkok Thailand were analyzed Phylogenetic and likelihoodmapping analyses of these genome sequences were performed Based on our results the starlike signal and topology of 2019nCoV may be indicative of potentially large first generation humantohuman virus transmission We estimated that 2019nCoV likely originated in Wuhan on 9 November 2019 95 credible interval 25 September 2019 and 19 December 2019 and that Wuhan is the major hub for the spread of the 2019nCoV outbreak in China and elsewhere Our results could be useful for designing effective prevention strategies for 2019nCoV in China and beyond This article is protected by copyright All rights reserved</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">An indepth annotation of the newly discovered coronavirus 2019nCoV genome has revealed <span class='answer'>differences between 2019nCoV and severe acute respiratory syndrome SARS or SARSlike coronaviruses A systematic comparison identified 380 amino acid substitutions</span> between these coronaviruses which may have caused functional and pathogenic divergence of 2019nCoV</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>5. what effors are being done in asia to prevent further outbreaks?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">During the last century 3 influenza A pandemics have occurred and pandemic influenza will inevitably occur in the future Although the timing and severity of the next pandemic cannot be predicted the probability that a pandemic will occur has increased based on the current outbreaks of AH5N1 in Asia Europe and Africa Because of these widespread outbreaks the World Health Organization declared a phase 3 pandemic alert in the fall of 2005 <span class='answer'>Early detection is essential to prevent the spread of avian influenza Planning now can be achieved by integrating interventions</span> to ensure a prompt and effective response to a pandemic This article provides an overview of the current status of AH5N1 influenza worldwide and recommendations for the prevention and control of avian influenza should it emerge in humans in the United States</div>



```python
task = 4
display_single_task(task, all_tasks[task-1])
```


<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h1 class='task_title'>Task 4: What do we know about vaccines and therapeutics?</h1>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>1. what drugs or therapies are being investigated?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">The high cost of drug development and the narrow spectrum of coverage typically provided by directacting antivirals limit the scalability of this antiviral approach This review summarizes progress and challenges in the repurposing of approved kinase inhibitors as hosttargeted broadspectrum <span class='answer'>antiviral therapies</span></div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">The direct replication of influenza virus is not the only cause of harm to human health influenza infection leading to a hyperinflammatory immune response can also result in serious conditions So the treatment strategy for influenza needs to keep balance between antivirus and antiinflammation Herein we review the treatment strategies of <span class='answer'>antiinfluenza drugs and traditional Chinese medicines</span></div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Synthetic biology is bringing together engineers and biologists to design and build novel biomolecular components networks and pathways and to use these constructs to rewire and reprogram organisms These reengineered organisms will change our lives in the coming years leading to cheaper drugs green means to fuel our cars and <span class='answer'>targeted therapies</span> to attack superbugs and diseases such as cancer The de novo engineering of genetic circuits biological modules and synthetic pathways is beginning to address these critical problems and is being used in related practical applications</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">BACKGROUND Influenza viruses cause serious infections that can be prevented or treated using <span class='answer'>vaccines or antiviral agents</span> respectively While vaccines are effective they have a number of limitations and influenza strains resistant to currently available antiinfluenza drugs are increasingly isolated This necessitates the exploration of novel antiinfluenza therapies METHODOLOGYPRINCIPAL FINDINGS We investigated the potential of aurintricarboxylic acid ATA a potent inhibitor of nucleic acid processing enzymes to protect MadinDarby canine kidney cells from influenza infection We found by neutral red assay that ATA was protective and by RTPCR and ELISA respectively confirmed that ATA reduced viral replication and release Furthermore while pretreating cells with ATA failed to inhibit viral replication preincubation of virus with ATA effectively reduced viral titers suggesting that ATA may elicit its inhibitory effects by directly interacting with the virus Electron microscopy revealed that ATA induced viral aggregation at the cell surface prompting us to determine if ATA could inhibit neuraminidase ATA was found to compromise the activities of virusderived and recombinant neuraminidase Moreover an oseltamivirresistant H1N1 strain with H274Y was also found to be sensitive to ATA Finally we observed additive protective value when infected cells were simultaneously treated with ATA and amantadine hydrochloride an antiinfluenza drug that inhibits M2ion channels of influenza A virus CONCLUSIONSSIGNIFICANCE Collectively these data suggest that ATA is a potent antiinfluenza agent by directly inhibiting the neuraminidase and could be a more effective antiviral compound when used in combination with amantadine hydrochloride</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>2. are anti-inflammatory drugs recommended?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">We report an evaluation of the cytotoxicity of a series of electrondeficient 16electron halfsandwich precious metal complexes of ruthenium osmium and iridium [OsRu6pcymene12dicarbaclosododecarborane12dithiolato] 12 [Ir5pentamethylcyclopentadiene12dicarbaclosododecarborane12dithiolato] 3 [OsRu6pcymenebenzene12dithiolato] 45 and [Ir5pentamethylcyclopentadienebenzene12dithiolato] 6 towards RAW 2647 murine macrophages and MRC5 fibroblast cells Complexes 3 and 6 were found to be <span class='answer'>noncytotoxic</span> The antiinflammatory activity of 16 was evaluated in both cell lines after nitric oxide NO production and inflammation response induced by bacterial endotoxin lipopolysaccharide LPS as the stimulus All metal complexes were shown to exhibit dosedependent inhibitory effects on LPSinduced NO production on both cell lines Remarkably the two iridium complexes 3 and 6 trigger a full antiinflammatory response against LPSinduced NO production which opens up new avenues for the development of noncytotoxic antiinflammatory drug candidates with distinct structures and solution chemistry from that of organic drugs and as such with potential novel mechanisms of action</div>



```python
task = 5
display_single_task(task, all_tasks[task-1])
```


<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h1 class='task_title'>Task 5: What do we know about non-pharmaceutical interventions?</h1>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>1. which non-pharmaceutical interventions limit tramsission?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">This commentary argues that 100 years after the deadly Spanish flu the public health emergency communitys responses to much more limited pandemics and outbreaks demonstrate a critical shortage of personnel and resources Rather than relying on nonpharmaceutical interventions such as <span class='answer'>quarantine</span> the United States must reorder its health priorities to ensure adequate preparation for a largescale pandemic</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>2. what are most important barriers to compliance?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Background Infection control practice compliance is commonly monitored by measuring hand hygiene compliance The limitations of this approach were recognized in 1 acute health care organization that led to the development of an Infection Control Continuous Quality Improvement tool Methods The Pronovost cycle Barriers and Mitigation tool and Hexagon framework were used to review the existing monitoring system and develop a quality improvement data collection tool that considered the context of care delivery Results Barriers and opportunities for improvement including <span class='answer'>ambiguity consistency and feasibility of expectations</span> the environment knowledge and education were combined in a monitoring tool that was piloted and modified in response to feedback Local adaptations enabled staff to prioritize and monitor issues important in their own workplace The tool replaced the previous system and was positively evaluated by auditors Challenges included ensuring staff had time to train in use of the tool time to collect the audit and the reporting of low scores that conflicted with a targetbased performance system Conclusions Hand hygiene compliance monitoring alone misses other important aspects of infection control compliance A continuous quality improvement tool was developed reflecting specific organizational needs that could be transferred or adapted to other organizations</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Background We aimed to investigate the frequency of standard precautions SPs compliance and the factors affecting the compliance among nursing students NSs Methods A crosssectional survey study guided by the health belief model was conducted in 2009 The study questionnaire is valid content validity index 081 and reliable Cronbach  range 065094 Results There were 678 questionnaires analyzed with a response rate of 689 The mean frequency score of SPs compliance was 438  040 out of 5 Tukey honest significant difference post hoc test indicated that year 2 and year 4 students had better SPs compliance than year 3 students Further analysis using a univariate general linear model identified an interaction effect of perceived influence of nursing staff and year of study F1593  372 P  05 The 5 following predictors for SPs compliance were identified <span class='answer'>knowledge of SPs perceived barriers adequacy of training management support and influence of nursing staff</span> Conclusion Although the SPs compliance among NSs was high the compliance varied by year of study and was affected by the nursing staff Furthermore SPs compliance among NSs can be enhanced by increasing SPs knowledge providing more SPs training promoting management support reducing identified SPs barriers and improving nursing staff compliance to SPs</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Summary Objective For reasons that have yet to be elucidated the uptake of preventive measures against infectious diseases by Hajj pilgrims is variable The aim of this study was to identify the preventive advice and interventions received by Australian pilgrims before Hajj and the barriers to and facilitators of their use during Hajj Methods Two crosssectional surveys of Australians pilgrims aged 18 years were undertaken one before and one after the Hajj 2014 Results Of 356 pilgrims who completed the survey response rate 94 80 had the influenza vaccine 30 the pneumococcal vaccine and 30 the pertussis vaccine Concern about contracting disease at Hajj was the most cited reason for vaccination 734 and not being aware of vaccine availability was the main reason for nonreceipt 56 Those who obtained pretravel advice were twice as likely to be vaccinated as those who did not seek advice Of 150 pilgrims surveyed upon return 94 reported practicing hand hygiene during Hajj citing ease of use 67 and <span class='answer'>belief in its effectiveness 624</span> as the main reasons for compliance university education was a significant predictor of hand hygiene adherence Fiftythree percent used facemasks with breathing discomfort 76 and a feeling of suffocation 40 being the main obstacles to compliance Conclusion This study indicates that there are significant opportunities to improve awareness among Australian Hajj pilgrims about the importance of using preventive health measures</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Background Health care workers compliance with infection control practices and principles is vital in preventing the spread of disease One tool to assess infection control practice in clinical areas is the infection control audit however many institutions do not approach this in a systematic fashion Methods Key features of the infection control audit were identified by the infection control team and developed into a standardized format for review of clinical areas The audit incorporates a review of the physical layout protocols and policies knowledge of basic infection control principles and workplace practice review Results Over the last 13 years the infection control unit has completed 17 audits involving 1525 employees Fourhundredone staff members have filled out questionnaires that assessed their understanding of standard precautions A total of 257 recommendations have been made and 95 of these have been implemented The majority of recommendations address <span class='answer'>separation of clean and dirty supplies hand hygiene compliance hand hygiene signage proper use of barriers and environmental cleaning</span> Conclusion The infection control audit is an opportunity to implement changes and to introduce remedial measures in collaboration with various departments and services A standardized approach to the audit allows benchmarking of practices across the institution and enhances standards of care</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer"> Influenza vaccination coverage among healthcare workers HCWs remains the lowest compared with other priority groups for immunization Little is known about the acceptability and compliance with the pandemic H1N1 2009 influenza vaccine among HCWs during the current campaign Between 23 December 2009 and 13 January 2010 once the workplace vaccination program was over we conducted a crosssectional questionnairebased survey at the University Hospital 12 de Octubre Madrid Spain Five hundred twentyseven HCWs were asked about their influenza immunization history during the 20092010 season as well as the reasons for accepting or declining either the seasonal or pandemic vaccines Multiple logisticregression analysis was preformed to identify variables associated with immunization acceptance A total of 262 HCWs 497 reported having received the seasonal vaccine while only 87 165 affirmed having received the pandemic influenza H1N1 2009 vaccine <span class='answer'>Selfprotection and protection of the patient</span> were the most frequently adduced reasons for acceptance of the pandemic vaccination whereas the existence of doubts about vaccine efficacy and fear of adverse reactions were the main arguments for refusal Simultaneous receipt of the seasonal vaccine odds ratio [OR] 027 95 confidence interval [95 CI] 014052 and being a staff OR 008 95 CI 004019 or a resident physician OR 016 95 CI 005050 emerged as independent predictors for pandemic vaccine acceptance whereas selfreported membership of a priority group was associated with refusal OR 598 95 CI 135265 The pandemic H1N1 2009 influenza vaccination coverage among the HCWs in our institution was very low 165 suggesting the role of specific attitudinal barriers and misconceptions about immunization in a global pandemic scenario</div>



```python
task = 6
display_single_task(task, all_tasks[task-1])
```


<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h1 class='task_title'>Task 6: What has been published about medical care?</h1>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>1. how does extracorporeal membrane oxygenation affect 2019-ncov patients?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>2. what telemedicine and cybercare methods are most effective?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer"><span class='answer'>Virtually Perfect Telemedicine</span> for Covid19 Telemedicines payment and regulatory structures licensing credentialing and implementation take time to work through but health systems that have a</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>3. how is artificial intelligence being used in real time health delivery?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>4. what adjunctive or supportive methods can help patients?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer"> Some newly emerging viral lung infections have the potential to cause large outbreaks of severe respiratory disease amongst humans In this contribution we discuss infections by influenza A H5N1 SARS and Hanta virus The H5N1 subtype of avian influenza bird flu has crossed the species barrier and causes severe illness in humans So far 328 humans in twelve countries have contracted the disease and 200 have died The young are particularly affected Oseltamivir is the antiviral drug of choice and should be given as early as possible Patients require supportive care often including invasive ventilation If H5N1 develops the ability to transmit efficiently between humans an influenza pandemic is likely Severe acute respiratory syndrome SARS was first seen in China in 2002 The outbreak was finally contained in 2003 by which time 8098 probable SARS cases had been identified with at least 774 deaths The virus was identified in 2003 as belonging to the coronaviridae family SARS is transmitted between humans and clusters have been seen The mainstay of treatment is supportive Various antiviral agents and <span class='answer'>adjunctive therapies</span> were tried but none were conclusively effective Hanta virus is an emerging cause of viral lung disease In 1993 a new species of Hanta virus was recognized after an outbreak of a new rapidly progressive pulmonary syndrome in the US 465 cases of Sin Nombre virus have now been seen in the US with a mortality rate of 35 Many of the confirmed cases had contact with rodents the major host of hanta viruses Treatment is supportive as there is no specific therapy</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer"> A severe inflammatory immune response with hypercytokinemia occurs in patients hospitalized with severe influenza such as avian influenza AH5N1 AH7N9 and seasonal AH1N1pdm09 virus infections The role of immunomodulatory therapy is unclear as there have been limited published data based on randomized controlled trials RCTs <span class='answer'>Passive immunotherapy</span> such as convalescent plasma and hyperimmune globulin have some studies demonstrating benefit when administered as an adjunctive therapy for severe influenza Triple combination of oseltamivir clarithromycin and naproxen for severe influenza has one study supporting its use and confirmatory studies would be of great interest Likewise confirmatory studies of sirolimus without concomitant corticosteroid therapy should be explored as a research priority Other agents with potential immunomodulating effects including nonimmune intravenous immunoglobulin Nacetylcysteine acute use of statins macrolides pamidronate nitazoxanide chloroquine antiC5a antibody interferons human mesenchymal stromal cells mycophenolic acid peroxisome proliferatoractivated receptors agonists nonsteroidal antiinflammatory agents mesalazine herbal medicine and the role of plasmapheresis and hemoperfusion as rescue therapy have supportive preclinical or observational clinical data and deserve more investigation preferably by RCTs Systemic corticosteroids administered in high dose may increase the risk of mortality and morbidity in patients with severe influenza and should not be used while the clinical utility of low dose systemic corticosteroids requires further investigation</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">As of 22 February 2020 more than 77662 cases of confirmed COVID19 have been documented globally with over 2360 deaths Common presentations of confirmed cases include fever fatigue dry cough upper airway congestion sputum production shortness of breath myalgiaarthralgia with lymphopenia prolonged prothrombin time elevated Creactive protein and elevated lactate dehydrogenase The reported severecritical case ratio is approximately 710 and median time to intensive care admission is 95105 days with mortality of around 12 varied geographically Similar to outbreaks of other newly identified virus there is no proven regimen from conventional medicine and most reports managed the patients with lopinavirritonavir ribavirin betainterferon glucocorticoid and supportive treatment with <span class='answer'>remdesivir</span> undergoing clinical trial In China Chinese medicine is proposed as a treatment option by national and provincial guidelines with substantial utilization We reviewed the latest national and provincial clinical guidelines retrospective cohort studies and case series regarding the treatment of COVID19 by addon Chinese medicine We have also reviewed the clinical evidence generated from SARS and H1N1 management with hypothesized mechanisms and latest in silico findings to identify candidate Chinese medicines for the consideration of possible trials and management Given the paucity of strongly evidencebased regimens the available data suggest that Chinese medicine could be considered as an adjunctive therapeutic option in the management of COVID19</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer"> Neonatal pneumonia may occur in isolation or as one component of a larger infectious process Bacteria viruses fungi and parasites are all potential causes of neonatal pneumonia and may be transmitted vertically from the mother or acquired from the postnatal environment The patients age at the time of disease onset may help narrow the differential diagnosis as different pathogens are associated with congenital earlyonset and lateonset pneumonia <span class='answer'>Supportive care and rationally selected antimicrobial therapy</span> are the mainstays of treatment for neonatal pneumonia The challenges involved in microbiological testing of the lower airways may prevent definitive identification of a causative organism In this case secondary data must guide selection of empiric therapy and the response to treatment must be closely monitored</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">A tissue culture micromethod is described for adenovirus isolation and preparation for presumptive identification by electron microscopy These procedures are easier more economical and faster than conventional methods The micro techniques make it more feasible to utilize direct visualization of virus in infected cells as an <span class='answer'>adjunctive diagnostic and research tool</span></div>



```python
task = 7
display_single_task(task, all_tasks[task-1])
```


<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h1 class='task_title'>Task 7: What do we know about diagnostics and surveillance?</h1>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>1. what diagnostic tests (tools) exist or are being developed to detect 2019-ncov?</h2>



```python
task = 8
display_single_task(task, all_tasks[task-1])
```


<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h1 class='task_title'>Task 8: Other interesting questions</h1>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>1. what is the immune system response to 2019-ncov?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Children comprise a special population whose immune response system is <span class='answer'>distinct from adults</span> Therefore pediatric patients infected with 2019nCoV have their own clinical features and therapeutic responses Herein we formulate this recommendation for diagnosis and treatment of 2019nCoV infection in children which is of paramount importance for clinical practiceSN  18670687</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">The 2019 novel coronavirus 2019nCoV infection has spread throughout China since the first case was identified in Wuhan Hubei Province in December 2019 According to previous knowledge and experience women during pregnancy and puerperium are a vulnerable population due to <span class='answer'>physiological changes</span> in their immune and cardiopulmonary system so making them more susceptible to viral infections Based on the latest 2019nCoV national management plan we propose this detailed plan of care to provide better prevention and management of 2019nCoV infection in women during pregnancy and the puerperium</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">2019 novel coronavirus 2019nCoV infection has been spreading in China since December 2019 Neonates are presumably the highrisk population susceptible to 2019nCoV due to <span class='answer'>immature immune function</span> The neonatal intensive care unit NICU should be prepared for 2019nCoV infections as far as possible The emergency response plan enables the efficient response capability of NICU During the epidemic of 2019nCoV the emergency response plan for the NICU should be based on the actual situation including diagnosis isolation and treatment as well as available equipment and staffing and take into account the psychosocial needs of the families and neonatal care staff</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Since the end of 2019 an outbreak of pneumonia caused by a novel coronavirus named 2019 novel coronavirus 2019nCoV has occurred in in China The dramatically rapid spread and strong infectivity of this virus has attracted global attention Neonates are thought to be susceptible to the virus because their immune system is <span class='answer'>not well developed</span> Neonates have been reported to be affected by this virus The Chinese Medical Association Chinese Medical Doctor Association Pediatric Professional Committee of the Chinese Peoples Liberation Army have put forward strategies for the effective prevention and control of the 2019nCoV infection in neonates This expert review summarized the key points of the above three prevention and control consensus and programs</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">Since December 2019 there has been an outbreak of novel coronavirus 2019nCoV infection in China Two cases of neonates with positive 2019nCoV tests have been reported Due to the <span class='answer'>immature</span> immune system and the possibility of vertical transmission from mother to infant neonates have become a highrisk group susceptible to 2019nCoV which emphasize a close cooperation from both perinatal and neonatal pediatrics In neonatal intensive care unit NICU to prevent and control infection there should be practical measures to ensure the optimal management of children potentially to be infected According to the latest 2019nCoV national management plan and the actual situation the Chinese Neonatal 2019nCoV expert working Group has put forward measures on the prevention and control of neonatal 2019nCoV infection</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>2. can personal protective equipment prevent the transmission of 2019-ncov?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">In late December 2019 a previous unidentified coronavirus currently named as the 2019 novel coronavirus 2019nCoV emerged from Wuhan China and resulted in a formidable outbreak in many cities in China and expanding globally including Thailand Republic of Korea Japan USA Philippines Viet Nam and our country as of 262020 at least 25 countries The disease is officially named as the Severe Specific Contagious Pneumonia SSCP in 1152019 and is a notifiable communicable disease of the 5 category by the Taiwan CDC the Ministry of Health SSCP is a potential zoonotic disease with low to moderate estimated 25 mortality rate Persontoperson transmission may occur through droplet or contact transmission and jeopardized firstline healthcare workers if lack of stringent infection control or <span class='answer'>no proper personal protective equipment available</span> Currently there is no definite treatment for SSCP although some drugs are under investigation To promptly identify patients and prevent further spreading physicians should be aware of travel or contact history for patients with compatible symptoms</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">A global health emergency has been declared by the World Health Organization as the 2019nCoV outbreak spreads across the world with confirmed patients in Canada Patients infected with 2019nCoV are at risk for developing respiratory failure and requiring admission to critical care units While providing optimal treatment for these patients careful execution of infection control measures is necessary to prevent nosocomial transmission to other patients and to healthcare workers providing care Although the exact mechanisms of transmission are currently unclear humantohuman transmission can occur and the risk of airborne spread during aerosolgenerating medical procedures remains a concern in specific circumstances This paper summarizes important considerations regarding <span class='answer'>patient screening environmental controls personal protective equipment resuscitation measures</span> including intubation and critical care unit operations planning as we prepare for the possibility of new imported cases or local outbreaks of 2019nCoV Although understanding of the 2019nCoV virus is evolving lessons learned from prior infectious disease challenges such as Severe Acute Respiratory Syndrome will hopefully improve our state of readiness regardless of the number of cases we eventually manage in Canada</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">In late December 2019 a previous unidentified coronavirus currently named as the 2019 novel coronavirus emerged from Wuhan China and resulted in a formidable outbreak in many cities in China and expanded globally including Thailand Republic of Korea Japan United States Philippines Viet Nam and our country as of 262020 at least 25 countries The disease is officially named as Coronavirus Disease2019 COVID19 by WHO on February 11 2020 It is also named as Severe Pneumonia with Novel Pathogens on January 15 2019 by the Taiwan CDC the Ministry of Health and is a notifiable communicable disease of the fifth category COVID19 is a potential zoonotic disease with low to moderate estimated 25 mortality rate Persontoperson transmission may occur through droplet or contact transmission and if there is a lack of stringent infection control or <span class='answer'>if no proper personal protective equipment available it may jeopardize the firstline healthcare workers</span> Currently there is no definite treatment for COVID19 although some drugs are under investigation To promptly identify patients and prevent further spreading physicians should be aware of the travel or contact history of the patient with compatible symptoms</div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><h2 class='question_title'>3. can 2019-ncov infect patients a second time?</h2>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer"><span class='answer'>20191220192019nCoV 2019</span>nCoV 2019nCoV </div>



<style>
        div {
            color: black;
        }

        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }

        .answer{
            color: #dc7b15;
        }

        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }

        div.output_scroll { 
            height: auto; 
        }

    </style><div class="single_answer">2002SARS2012MERSCoV<span class='answer'>2019</span> 2019nCovCoV2019nCov 2019nCov 2019nCov</div>


### Export Results


```python
output_path = data_dir / "covid_kaggle_answer_from_biobert_qa.json"
```


```python
import json
with open(output_path, "w") as f:
    json.dump(all_answers, f)
```
