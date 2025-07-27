# Text-classification-pipeline
This project focuses on building an end-to-end text classification pipeline to distinguish between real and fake disaster-related tweets. It also includes an exploration of different text representation techniques and a simple generative model.

Here are the key steps performed in the notebook:

1.  **Required Imports:** Import necessary libraries for data manipulation, visualization, natural language processing, and machine learning (pandas, numpy, matplotlib, seaborn, nltk, sklearn, gensim).
2.  **Loading Dataset:** Load the training and testing datasets from CSV files into pandas DataFrames.
3.  **Data Inspection:** Check the shape of the DataFrames and display the head to understand the data structure.
4.  **Class Distribution Check:** Analyze the distribution of the target variable (real vs. fake tweets) in both training and testing datasets.
5.  **Preprocessing Function:** Define a function to clean and preprocess the raw text data. This includes converting to lowercase, removing URLs, mentions, hashtags, and special characters, tokenizing, removing stopwords, and lemmatizing.
6.  **Applying Preprocessing:** Apply the preprocessing function to the 'text' column of both training and testing DataFrames to create a new 'clean_text' column.
7.  **Setting Up Features and Labels:** Define the features (`X_train`, `X_test` using the 'clean_text' column) and labels (`y_train`, `y_test` using the 'target' column).
8.  **Feature Engineering - Bag of Words (BoW):** Use `CountVectorizer` to create a Bag of Words representation of the cleaned text data.
9.  9.  **Feature Engineering - TF-IDF (bi-grams):** Use `TfidfVectorizer` with `ngram_range=(1, 2)` to create a TF-IDF representation considering unigrams and bigrams.
10. **Word2Vec Averaging:**
    *   Install the `gensim` library.
    *   Tokenize the cleaned text data using `gensim.utils.simple_preprocess`.
    *   Train a `Word2Vec` model on the training data tokens.
    *   Define a function to calculate the average Word2Vec vector for a list of tokens.
    *   Apply the averaging function to create dense Word2Vec representations for both training and testing data.
11. **Modeling & Evaluation:**
    *   Import necessary modules for Naive Bayes (`MultinomialNB`), Logistic Regression (`LogisticRegression`), and evaluation metrics (`classification_report`, `accuracy_score`).
    *   Define a function `evaluate` to train a given model on provided data and print accuracy and classification report.
    *   Evaluate Naive Bayes models using both BoW and TF-IDF features.
    *   Evaluate Logistic Regression models using BoW, TF-IDF, and averaged Word2Vec features.
12. **Markov Chain Text Generation (Character 3-gram):**
    *   Define a function `build_markov_chain` to create a character-based Markov chain (3-gram) from the training text.
    *   Define a function `generate_text` to generate new text samples using the built Markov chain.
    *   Build the Markov chain model from the training text.
    *   Generate and print sample text outputs from the Markov chain.
