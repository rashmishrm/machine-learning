from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


with open("dataset/imdb_labelled.txt","r") as text_file:
    lines = text_file.read().split("\n")
with open("dataset/yelp_labelled.txt","r") as text_file:
    lines += text_file.read().split("\n")
with open("dataset/amazon_cells_labelled.txt","r") as text_file:
    lines += text_file.read().split("\n")

    lines= [line.split("\t") for line in lines if ((len(line.split("\t"))==2) and (line.split("\t")[1]<> ''))]
    training_documents = [line[0] for line in lines]

    training_labels = [line[1] for line in lines]

    count_vectorizer = CountVectorizer(binary='true')
training_documents = count_vectorizer.fit_transform(training_documents)

classfier= BernoulliNB().fit(training_documents,training_labels)
print classfier.predict(count_vectorizer.transform(["this is the best movie"]))
print classfier.predict(count_vectorizer.transform(["this is the bad movie"]))
print classfier.predict(count_vectorizer.transform(["this is the fantastic movie"]))
