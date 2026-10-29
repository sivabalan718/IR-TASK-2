TEXT CLASSIFICATION BATTLE
Task 2 --- Compare, Analyze and Defend
Team Member 1: Siva Balan S
Team Member 2: Shailesh M

Course: Information Retrieval
Institution: SRM Institute of Science and Technology
1. Project Overview
This project builds and compares machine-learning models for classifying
news articles into predefined categories using the BBC News dataset. The
notebook covers data inspection and cleaning, TF-IDF feature extraction,
model training, evaluation, model comparison, and a KNN parameter
experiment.
2. Objectives
- Prepare the news dataset for machine learning.
- Convert article text into numerical features using TF-IDF.
- Train four classification algorithms.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Compare performance using a table, graph, and confusion matrices.
- Investigate how changing the number of neighbours affects KNN
  performance.
3. Dataset
Dataset: BBC News
File: BBC News Train.csv (the uploaded filename may differ)

  Column        Description
  ArticleId   Unique identifier for an article
  Text        News article text used as input
  Category    Target label to predict

The five categories are Business, Entertainment, Politics, Sport, and
Tech.

The recorded run contained 1,490 rows and used an 80:20 stratified
train-test split with random_state=42: 1,192 training samples and
298 test samples.
4. Data Preparation and TF-IDF
The recorded workflow: 1. Removes duplicate article IDs. 2. Removes rows
with missing text or category values. 3. Removes text entries that are
empty after trimming whitespace. 4. Separates Text (input) from
Category (target). 5. Splits the data using stratification. 6.
Converts text into numerical vectors using TF-IDF (Term
Frequency--Inverse Document Frequency).

Recorded TF-IDF settings: - lowercase=True - stop_words='english' -
max_features=5000 - ngram_range=(1, 1) --- unigrams - min_df=2

The vectorizer is fitted on training text and then applied to test text
to help prevent test-data leakage.
5. Machine-Learning Models
  Model                               Recorded configuration
  Decision Tree                       DecisionTreeClassifier(random_state=42)

  K-Nearest Neighbours (KNN)          KNeighborsClassifier(n_neighbors=5, metric='cosine', algorithm='brute')

  Support Vector Machine (SVM)        LinearSVC(random_state=42, max_iter=5000)
  Multinomial Naive Bayes             MultinomialNB(alpha=1.0)
6. Evaluation Metrics
- Accuracy: Overall proportion of correct predictions.
- Precision: How often predictions for a category are correct.
- Recall: How many actual examples of a category are correctly
  identified.
- Macro F1-score: The unweighted average of F1-scores across
  categories.
Confusion matrices are used to inspect correct and incorrect predictions
by category.
7. Recorded Model Results
  Model                       Accuracy   Macro Precision   Macro Recall   Macro F1-score
  Decision Tree                 82.89%            82.72%         82.21%           82.40%
  KNN (K=5)                     92.28%            91.95%         91.95%           91.93%
  Linear SVM                    97.65%            97.64%         97.50%           97.55%
  Multinomial Naive Bayes       97.65%            97.61%         97.54%           97.57%
Result interpretation
- Linear SVM and Multinomial Naive Bayes achieved the highest recorded
  accuracy: 97.65%.
- Multinomial Naive Bayes achieved the highest recorded macro
  F1-score: 97.57%.
- KNN outperformed the Decision Tree in this experiment.
- Results apply to the recorded split and settings; they may differ if
  the dataset or configuration changes.
8. What-If Experiment: Changing K in KNN
KNN was tested with different neighbour counts while retaining cosine
distance and brute-force search.
 K   Accuracy   Macro F1-score
 3     92.62%           92.34%
 5     92.28%           91.93%
 7     92.95%           92.74%
 9     93.62%           93.35%
11     93.62%           93.41%
Observation: K=9 and K=11 tied for the highest recorded accuracy.
K=11 produced the highest recorded macro F1-score among the tested
values.
9. Conclusion
This project demonstrates a text-classification workflow using TF-IDF
and four machine-learning algorithms. In the recorded experiment, Linear
SVM and Multinomial Naive Bayes achieved the strongest overall results.
The KNN experiment showed that changing the number of neighbours
affected performance. Model choice should be based on relevant
evaluation metrics rather than accuracy alone.
10. How to Run
1. Open IR-TASK 2.ipynb in Google Colab or Jupyter Notebook.
2. Upload the BBC News CSV file when prompted, or update the dataset
   path to match its location.
3. Run the notebook cells from top to bottom.
4. Review the comparison table, confusion matrices, and KNN experiment
   outputs.
Requirements: Python 3, pandas, scikit-learn, matplotlib, and Google
Colab or Jupyter Notebook.
11. Repository
GitHub repository:
https://github.com/sivabalan718/text-classification-battle
