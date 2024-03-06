# native-bayes-classifier
Created a native bayes classifier for a Computational Linguistics class activity, using python. 
This classifier is trained on 5 files with each being one of 3 genres, "info", "novel" and "soap", tokenizing the files into lower case alphanumeric features that also include dashes, excluding "banned" words from stopwords.txt. Features that appear less that 10 times in a file are also ignored.
Writes the model created to a file called nb.params, which includes the class (genre) priors and the likelihood of each feature within each class. 
This model is then used to classify a test file into one of the three given genres, printing the top 15 features used to recognize and categorize the test file, for each genre. The genre yielding the highest accumulated probability value is the chosen genre. 

>>> train-script.py info-1 info-2 novel-1 soap-1 soap-2 #trains the naive classifier on the 5 files
>>> test-script.py nb.params test-1 #classifies the test file
