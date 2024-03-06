import sys, re, math

#declare relevant dictionaries that will store variables required to train our naive Bayes classifier.
vocab = {}
ndoc = len(sys.argv) - 1
classes = {}
bigdoc = {}
loglikelihood = {}

#[classify] reads each training file from the list of training files in sys.argv.
#tokenizes the words read by alphanumeric characters and dashes, in lowercase, 
##eliminating words in stopwords.txt, discarding words that appear less that 10 times. 
#populates two dimensional array [bigdoc], which contains n main dictionaries 
##separating each nth genre of files, with each sub-dictionary containing all the loaded 
##words from a file of that genre. 
#Dictionary [vocab] is an accumulative dictionary containing all the loaded files, regardless of genre. 

def classify() :
    i = 1
    while i < len(sys.argv) :
        with open(sys.argv[i], "r", encoding="utf-8", errors="ignore") as in_f :
            classified = in_f.name.split("-")[0]
            for line in in_f :
                split_line = line.strip().split()
                for x in split_line : 
                    lowerx = x.lower()
                    if re.match ('^[a-zA-Z0-9\\-]+$', lowerx):
                        with open('stopwords.txt') as file:
                                content = file.read().split()
                                if lowerx not in content :
                                    if lowerx in vocab :
                                        vocab[lowerx] += 1
                                        if classified not in bigdoc :
                                            bigdoc[classified] = {}
                                            bigdoc[classified][lowerx] = 1
                                        else : 
                                            if lowerx not in bigdoc[classified]:
                                                bigdoc[classified][lowerx] = 1
                                            else :
                                                bigdoc[classified][lowerx] += 1
                                    else :
                                        vocab[lowerx] = 1
                                        if classified not in bigdoc :
                                            bigdoc[classified] = {}
                                            bigdoc[classified][lowerx] = 1
                                        else : 
                                            if lowerx not in bigdoc[classified]:
                                                bigdoc[classified][lowerx] = 1
                                            else :
                                                bigdoc[classified][lowerx] += 1
#
            if classified in classes.keys():
                classes[classified] += 1
            else :
                classes[classified] = 1
        i += 1
        
    vocabmaker = list(vocab.keys())

    for key in vocabmaker:
        if vocab[key] < 10:
            vocab.pop(key, None)
    
    bigdocmaker = list(bigdoc.keys())

    for key in bigdocmaker:
        listmaker = list(bigdoc[key].keys())
        for k in listmaker:
            if k not in vocab:
                del bigdoc[key][k]

    update_to_log_prior()

#helper method for classify, updates the dictionary [classes] to the logprior values for each genre(class)
def update_to_log_prior():
    for c in classes:
        nc = classes[c] 
        classes[c] = math.log2(nc/ndoc)
        print(str(nc) + " " + str(ndoc) + " " + str(classes[c]))

#builds the naive Bayes classifier, using [bigdoc] and [vocab], to calculate the likelihood 
##of each word appearing in texts of a certain genre. 
def train_naive_bayes() :
    for c in classes:
        loglikelihood[c] = {}
        for w in vocab:
            if w in bigdoc[c]:
                count = bigdoc[c][w]
            else: count = 0
            summation = 0 
            for x in vocab:
                if x in bigdoc[c]:
                    summation += bigdoc[c][x]
            loglikelihood[c][w] = math.log2((count+1)/(summation + len(vocab))) 

#writes the model to a file named nb.params, printing the prior probability of each genre, followed by 
##the likelihood of all characteristic features of that genre appearing.
            
def write(): 
    classify()
    train_naive_bayes()
    with open("nb.params", "w") as file:
        for k, v in classes.items():
            file.write(str(k) + " PRIOR: " + str(v)+ "\n")
            for key, val in loglikelihood[k].items():
                file.write(str(k) + " " + str(key) + " : " + str(val) + "\n")
    
        
if __name__ == "__main__":
    write()