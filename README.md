##Ac Project
***In this project, we are implementing serial and parallel code of feature selection.
We used kmenas before feature selection.
What kind of analysis will get in the following project?
How much time and accuracy?

Analysis of accuracy and time consumption in the following methods:
1. Direct Classifier
2. Feature Selection Method + Classifier
3. Kmeans Serial + Feature Selection Method + Classifier
4. Kmenas Parallel + Feature Selection Method + Classifier
In the above project 
Classifier: Naive Bayes
Feature Selection:  Variance Inflation Factor (VIF)***


##How to run code?
python3 main.py

In this main.py file, need to edit the following parameters
file:= Path to CSV dataset
text_indices:= This list is consisting of column number which consists of TEXT data
start:= From which column index essential feature is starting, till the end.
out:= At which column index output field is stored (This is especially needed for measuring accuracy; for accuracy measure, we used Naive Bayes classifier which is supervised in nature)
