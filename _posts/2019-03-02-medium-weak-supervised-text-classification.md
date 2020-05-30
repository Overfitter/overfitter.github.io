---
layout: post
title: Text Classification Using Weakly Supervised — Deep Neural Networks
subtitle: Introduction to Weak Supervised Learning
# gh-repo: daattali/beautiful-jekyll
# gh-badge: [star, fork, follow]
tags: [nlp, text-classification, weak-supervised-learning, data-science, machine-learning]
# comments: true
---

## **What is Weakly-Supervised Learning?**

Many traditional lines of research in machine learning are similarly motivated by the insatiable appetite of modern machine learning models for labeled training data.

***Weak supervision is about leveraging higher-level and/or noisier input from subject matter experts (SMEs).***

In the **weak supervision learning**, our objective is the same as in supervised learning (where we have a massive set of labeled data). However, instead of a ground-truth labeled training set we have:

* Unlabeled data

* Some labeled data from the SME

* Learn a generative model over coverage and accuracy

There are multiple ways in which we can do weak — supervised learning. But the general scheme is that of a bootstrapping paradigm and is as follows:-

 1. **Initial Supervision **— seed examples for training an initial model or seed model itself

2. **Classification** — Classify unlabeled corpus with seed model

3. **Reinforcement** — Add most confident classifications to training and iterate

![Fig 1: Weakly — Supervised Learning Pipeline (Source: [http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds07.pdf](http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds07.pdf))](https://cdn-images-1.medium.com/max/3252/0*0-OpebHBuQNghvtI.png)

## **Weakly-Supervised Neural Text Classification (WEST)**

In this paper, they’ve proposed a weakly-supervised method that addresses the lack of training data in neural text classification.

This method consists of the following modules (Shown in Figure 2)

*(1) Pseudo-document generator that leverages seed information to generate pseudo-labeled documents for model pre-training*

*(2) Pre-training step: Training deep neural models (CNN/RNN) using word2vec(Skip-gram/BOW)/pre-trained embeddings features*

*(3) Self-training module that bootstraps on real unlabeled data for model refinement*

![Fig 2: WEST Pipeline (Source: [https://arxiv.org/pdf/1809.01478.pdf](https://arxiv.org/pdf/1809.01478.pdf))](https://cdn-images-1.medium.com/max/2464/1*8lRorqLNAQmcg4qobA6t8g.png)

Let's get deep dive into the above modules.

### **Psuedo Document Generation:**

Pseudo-document generator leverages seed information to generate pseudo-documents as synthesized training data. By assuming word and document representations reside in the same semantic space, we generate pseudo-documents for each class by modeling the semantics of each class as a high-dimensional spherical distribution (For Eg: Politics, Technology, Sports), and further sampling keywords to form pseudo-documents.

The pseudo document generator cannot only expand user-given seed information for better generalization, but also handle different types of seed information (e.g., label surface names, class-related keywords, or a few labeled documents) flexibly.

Specifically, we first use the Skip-Gram model to learn p dimensional vector representations of all the words in the corpus. Furthermore, since directional similarities between vectors are more effective in capturing semantic correlations, we normalize all the p-dimensional word embeddings so that they reside on a unit sphere, which is the joint semantic space. We call it “joint” because we assume pseudo-document vectors reside on the same unit sphere as well. We retrieve a set of keywords in the semantic space that are correlated to each class based on the seed information.

Let's understand the different types of seed information which we’ve discussed above:
>  **Label surface names: **When only label surface names (L) are given as seed information, for each class (j) we use the embedding of its surface name (L) to retrieve top-t nearest words in the semantic space. We set t to be the largest number that does not results in shared words across different classes.
>  **Class-related keywords: **When users provide a list of related keywords (S) for each class (j), we use the embeddings of these seed keywords to find top-t keywords in the semantic space, by measuring the average similarity to the seed keywords.
>  **Labeled documents:** When users provide a small number of documents (Dj) that are correlated with class (j), we first extract t representative keywords in Dj using tf-idf weighting, and then consider them as class-related keywords.

After obtaining a set of keywords that are correlated with each class, we model the semantic of each class as a [***von Mises-Fisher (vMF) distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution)***, which models word embeddings on a unit sphere.

### Neural Models With Self-Training:

A self-training module that fits real unlabeled documents for model refinement. First, the self-training module uses pseudo-documents to pre-train either CNN-based or RNN-based models to produce an initial model, which serves as a starting point in the subsequent model refining process. Then, it applies a self-training procedure, which iteratively makes predictions on real unlabeled documents and leverages high confidence predictions to refine the neural model.

Basically, in self-training, a model is used to correct its own mistakes by bootstrapping.

### Data Preparation:

In this paper, they’ve used two corpora from different domains to evaluate the performance of the proposed method: (The datasets are available on the [Github](https://github.com/yumeng5/WeSTClass))
>  (1) **AG’s News:** Training set portion (120, 000 documents evenly distributed into 4 classes) as the corpus for evaluation
>  (2) **Yelp Review**: Testing set portion (38, 000 documents evenly distributed into 2 classes) as the corpus for evaluation.

Here we’ll be using the XYZ data (new dataset) for the implementation.

 1. First of all, we’ve to prepare the initial training dataset (which we get from the SME) in below-mentioned format (**FileName: doc_id.txt**):
>  **0**:2340,12618,34618,6292,18465,7068,9944,36940,6038,20336,1205,23537,26182,3723,18540,10607,30545,11683,10583,10922
>  **1**:19409,26932,21657,23110,13393,4068,27462,12733,31707,6589,35955,31720,338,34263,20513,20702,36768,31479,9598,13392

**0** and **1 **are the two different classes of your XYZ Data and the other numbers (For Eg: 2340,12618) are the row index of the **dataset.csv** file of the different class.

*To generate the doc_id.txt file you can use the following code:*

![doc_id.txt file generation code](https://cdn-images-1.medium.com/max/2704/1*t9cJ2oRw1gsvpGCKzotj2w.png)

2. Now create the keywords.txt file in below-mentioned format (If you don’t have specific keywords in your dataset then you can skip this step and the algorithm will automatically generate the keywords from the docs using tf-idf technique as we’ve discussed above)

3. Also make sure you have the dataset.csv file in below format **(Column 1: Your Output, Column 2: Raw Text)**:

![dataset.csv file format (Source: Yelp Data)](https://cdn-images-1.medium.com/max/2244/1*xt6nzTWX6k9iuDZfX7EMOg.png)

### WEST Implementation:

After preparing the relevant files and dataset you’ve to make the following changes in the code to implement it on your XYZ dataset:

 1. **load_data.py:**

*Function name: read_file*

 <iframe src="https://medium.com/media/b144ecdb180ac9fb073f6841d264a301" frameborder=0></iframe>

*Function name: load_cnn & load_rnn*

 <iframe src="https://medium.com/media/61b11d44534a93a5b11ec1fbf7dd9a16" frameborder=0></iframe>

2. **main.py:**

After that, you’ve to make one more change in main.py file:

*parser.add_argument(‘ — dataset’, default=’agnews’, choices=[‘agnews’, ‘yelp’, ‘XYZ])*

Now you’re ready to run this code on your XYZ dataset by following below steps:

    python main.py --dataset ${dataset} --sup_source ${sup_source} --model ${model}

Replace : ${dataset}with XYZ, ${sup_source} (could be one of ['labels', 'keywords', 'docs'] and the type of neural model to use in ${model} (could be one of ['cnn', 'rnn'])

For Eg:

    python main.py --dataset XYZ --sup_source docs --model cnn

## **References:**

 1. [https://arxiv.org/pdf/1809.01478.pdf](https://arxiv.org/pdf/1809.01478.pdf)

 2. [https://github.com/yumeng5/WeSTClass](https://github.com/yumeng5/WeSTClass)

 3. [http://sujitpal.blogspot.com/2018/02/generating-labels-with-lu-learning-case.html](http://sujitpal.blogspot.com/2018/02/generating-labels-with-lu-learning-case.html)

Hope you’ve liked the article.

Thank You for reading :)!

Cheers!!
