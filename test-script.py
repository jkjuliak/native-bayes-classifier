import sys, math

#declare relevant dictionaries that will store variables required to test our naive Bayes classifier.
sum = {}
unlogsum = {}
loglikelihood = {}
testdoc = {}
proportions = {}

#reads the file containing the model, extracting relevant prior and likelihood probabilities, also 
##tokenizes the test file by whitespace, making all tokens lowercase.
#receives model file and test file from sys.argv
def read():
    with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as model:
        for line in model :
            split_line = line.strip().split()
            if "PRIOR" in split_line[1]:
                sum[split_line[0]] = float(split_line[2])
                unlogsum[split_line[0]] = 0
            else :
                if split_line[0] not in loglikelihood.keys(): 
                    loglikelihood[split_line[0]] = {}
                loglikelihood[split_line[0]][split_line[1]] = float(split_line[3])
    
    
    with open(sys.argv[2], "r", encoding="utf-8", errors="ignore") as testfile:
        index = 0
        for line in testfile :
            split = line.strip().split()
            for x in split:
                x = x.lower()
                testdoc[index] = x
                index += 1
    
    
#based on the tokens extracted from the test file, [test_native_bayes] accumulates the probability that the test file is from a certain genre.
#each accumulated sum is stored by genre in the dictionary [sum].
def test_naive_bayes():
    for c in sum.keys():
        for _ , x in testdoc.items():
            if x in loglikelihood[c].keys():
                sum[c] += loglikelihood[c][x]
                unlogsum[c] += math.pow(2, loglikelihood[c][x])
    

#when a token from the test file matches a feature from a certain genre in the model, [get_features] calculates this word's probability weight that contributes to the accumulated probability
## of the test file being from that genre. This information is stored in the dictionary [proportions], a nested dictionary whose subdictionaries are labelled by genre.
def get_features():
    for c in sum.keys():
        proportions[c] = {} 
        for _, x in testdoc.items():
            if x in loglikelihood[c].keys():
                if x not in proportions[c]:
                    proportions[c][x] = math.pow(2, loglikelihood[c][x]) / unlogsum[c]
                else:
                    proportions[c][x] += math.pow(2, loglikelihood[c][x]) / unlogsum[c]
                    
                

#prints the probabilities (not in logspace) of the top 15 features that contributed to the projected proabilities of each contending genre,
##the maximum of which is the genre predicted by the model. 
def print_features(): 
    for c in proportions.keys():
        print(c + " " + str(sum[c]))
        reverse = dict(sorted(proportions[c].items(), key=lambda x:x[1], reverse = True))
        index = 0 
        for key, value in reverse.items(): 
            if index < 15:
                print(c + " " + str(index) + ": " + key + " " + str(value))
                index += 1
        print("\n")

        
#prints the predicted class returned from the probabilities in the model, classifying the test file
def find_best_c():
     best_c = max(sum, key=sum.get)
     print("Class of test file: " + best_c)
     


if __name__ == "__main__":
    read()
    test_naive_bayes()
    get_features()
    find_best_c()
    print_features()