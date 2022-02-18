import utils
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import jellyfish   # for Levenshtein distance
import re
import math
import nltk

from nltk.tokenize import sent_tokenize   # for sentence tokenization
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# install nltk packages
nltk_downloader = nltk.downloader.Downloader()
if not nltk_downloader.is_installed('punkt'):
    nltk_downloader.download('punkt')
if not nltk_downloader.is_installed('averaged_perceptron_tagger'):
    nltk_downloader.download('averaged_perceptron_tagger')

####
# Stopwords list and tokenization functions
####

# nltk stoplist is not complete
nltk_sw = ['d', 'm', 'o', 's', 't', 'y', 'll', 're', 've', 'ma',
 "that'll", 'ain',
 "she's", "it's", "you're", "you've", "you'll", "you'd",
 'isn', "isn't", 'aren', "aren't", 'wasn', "wasn't", 'weren', "weren't",
 'don', "don't", 'doesn', "doesn't", 'didn', "didn't",
 'hasn', "hasn't", 'haven', "haven't", 'hadn', "hadn't",
 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
 'shan', "shan't", 'shouldn', "shouldn't", "should've",
 'won', "won't", 'wouldn', "wouldn't", 'couldn', "couldn't",
 'i', 'me', 'my', 'we', 'our', 'ours', 'you', 'your', 'yours',
 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs',
 'himself', 'herself', 'itself', 'myself', 
 'yourself', 'yourselves', 'ourselves', 'themselves',
 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
 'had', 'has', 'have', 'having', 'do', 'does', 'did', 'doing',
 'a', 'an', 'the', 'and', 'but', 'if', 'or',
 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
 'about', 'against', 'between', 'into', 'through', 
 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
 'over', 'under', 'here', 'there', 'when', 'where', 'why', 'how',
 'all', 'any', 'both', 'each', 'few', 'more', 'most',
 'other', 'some', 'such', 'no', 'nor', 
 'only', 'own', 'same', 'so', 'than', 'too', 
 'again', 'further', 'then', 'once', 'can', 'will', 'just', 
 'should', 'now']

# removed from nltk stoplist: not, very

added_sw = [ "he's", "he'd", "she'd", "he'll", "she'll", "you'll", 
            "they'd", "could've", "would've", 'could', 'would', "i'm", 'im',
           "thatll", "shes", "youre", "youve", "youll", "youd",
            "isnt", "arent", "wasnt", "werent",
            "dont", "doesnt", "didnt",
            "hasnt", "havent", "hadnt",
            "mightnt", "mustnt", "neednt", 
            "shant", "shouldnt", "shouldve",
            "wont", "wouldnt", "couldnt", 
            'a','b','c','e','f','g','h','i','j','k','l','n','p','q','r','u','v','w','x','z','lol']

stop_words = added_sw + nltk_sw

punc = ''',.;:?!'"()[]{}<>|\/@#^&*_~=+\n\t'''  #exclude hyphen, $, %
fullstop = '.'

# Input a string
# Returns a list of tokens with no stopwords, punctuation, numbers

def text_preprocess_clean(review):
    for p in punc:
        review = review.replace(p,' ')
    review = review.lower()
    review = review.replace('protectors','protector')
    review = review.replace('headphones','headphone')
    review = review.replace('iphones','iphone')
    review = review.replace('phones','phone')
    review = review.replace('mounts','mount')
    review = review.replace('stands','stand')
    review = review.replace('adapters','adapter')
    review = review.replace('chargers','charger')
    review = review.replace('cables','cable')
    review = review.replace('packs','pack')
    review = review.replace('batteries','battery')
    review = review.replace('cards','card')
    review = review.replace('styluses','stylus')
    review = review.replace('kits','kit')
    review = review.replace('speakers','speaker')
    review = review.replace('docks','dock')
    review = review.replace('boosters','booster')
    review = review.replace('cases','case')
    review = re.sub('\d+', '', review)
    review = word_tokenize(review)
    review = [w for w in review if w not in stop_words]
    
    return review

# Input a string
# Returns a list of tokens with punctuation and numbers and stopwords
# (punctuation allows us to eliminate meaningless bigrams containing punctuation symbols)

def text_preprocess(review):
    review = review.replace(fullstop,' . ')
    review = review.lower()
    review = review.replace("'m'",' am')
    review = review.replace("'s'",' is')
    review = review.replace("'re'",' are')
    review = review.replace("'ve'",' have')
    review = review.replace("'ll'",' wi11')
    review = review.replace("'d'",'')
    review = review.replace("n't",' not')
    review = review.replace("shan't",'shall not')
    review = review.replace("won't",'will not')
    review = review.replace('protectors','protector')
    review = review.replace('headphones','headphone')
    review = review.replace('phones','phone')
    review = review.replace('iphones','iphone')
    review = review.replace('mounts','mount')
    review = review.replace('stands','stand')
    review = review.replace('adapters','adapter')
    review = review.replace('chargers','charger')
    review = review.replace('cables','cable')
    review = review.replace('packs','pack')
    review = review.replace('batteries','battery')
    review = review.replace('cards','card')
    review = review.replace('styluses','stylus')
    review = review.replace('kits','kit')
    review = review.replace('speakers','speaker')
    review = review.replace('docks','dock')
    review = review.replace('boosters','booster')
    review = review.replace('cases','case')
    review = word_tokenize(review)

    return review


####
# Calculate idf for all words in the corpus, excluding stopwords
####

# Create the dictionary word_df (include stopwords)
# dataframe = df_allreview

def compute_idf(dataframe):
    df = dataframe

    vocabulary = set()               # corpus vocabulary including stopwords
    doc_f = defaultdict(lambda: 0)   # dictionary {word : num of products whose reviews contain the word (document frequency)}
    idf = dict()                     # dictionary {word : idf}
    
    for i, row in df.iterrows():
        t1 = text_preprocess_clean(row['all_reviews'])      
        vocabulary.update(t1)                     
    
        t2 = set(text_preprocess_clean(row['all_reviews']))  
        for t in t2:
            doc_f[t] += 1
    
    vocabulary = list(vocabulary)

    DOC_COUNT = len(df)                 # DOC COUNT = number of products (each product has an allreviews document)

    VOCAB_COUNT = len(vocabulary)      # number of unique words
    print(f'Number of words in corpus (excluding stopwords): {VOCAB_COUNT}')
    print(f'Number of documents (products): {DOC_COUNT}')
    
    # Calculate the idf of each word in the vocabulary
    for w in vocabulary:
        idf[w] = math.log10(DOC_COUNT / float(doc_f[w]))    # log to base 10
    
    return idf


####
# Search for and print the product's reviews
####

# search for the product with index idx and prints the data for the product
# dataf = df_review

def search(idx, dataf):
    pid = dataf.loc[idx]['asin']
    n = dataf.loc[idx]['num_reviews']
    
    print(f'Index: {idx}')
    print(f'Product ID: {pid}')
    print(f'Number of reviews: {n}\n')
    print('Sample reviews:\n')
    
    for i in range(1,4):
        rev = dataf.loc[idx][i+1]
        print(f'Review {i}:\n {rev}\n')
        
    return n

####
# Calculate tf and tf-idf for each word in the product's reviews (excluding stopwords)
####

# Returns a dictionary {word : tf) for all words (excluding stopwords) of the product
# tf = word frequency / total number of words (excluding stopwords)
# dataf = df_allreview

def word_tfidf(idx, idf, dataf):
  
    allrev = dataf.loc[idx]['all_reviews']    
    
    u1 = text_preprocess_clean(allrev)
    u2 = set(text_preprocess_clean(allrev))
    u2 = list(u2)
       
    tfreq = defaultdict(lambda: 0)   # {word : freq of word in all_reviews}
    tf = defaultdict(lambda: 0)
    tfidf = defaultdict(lambda: 0)
    
    for w in u1:
        tfreq[w] += 1                         

    for w in u2:
        tf[w] = 1 + math.log10(float(tfreq[w]))
        if w in idf:
            tfidf[w] = tf[w] * idf[w]
        else:
            tfidf[w] = tf[w] 
            
    return tfreq, tf, tfidf

####
# Get all candidate phrases (unigrams, bigrams and trigrams) by tokenization and filter out undesirable candidates
####

# returns all unigrams (excluding stopwords) for the product with index idx
# dataf = df_allreview

def unigram(idx, dataf):
    allrev = dataf.loc[idx]['all_reviews']  # type(allrev) = str
    
    u = set(text_preprocess_clean(allrev))   
    u = list(u)
    
    return u


# returns all bigrams for the product with index idx
# remove and reduce bigrams by checking punctuation and stopwords
# dataf = df_allreview

def bigram(idx, dataf):    
    allrev = dataf.loc[idx]['all_reviews']  
    u = text_preprocess(allrev)   
    b1 = set(nltk.ngrams(u, 2))  
    b1 = list(b1)
    
    b2 = []

    for b in b1:
        if (b[0] in stop_words) or (b[1] in stop_words) or (b[0] in punc) or (b[1] in punc):  
            continue

        if (b[0] not in punc) and (b[1] not in punc) and (b[0] not in stop_words) and (b[1] not in stop_words):
            b2.append(b)    
            
    b2 = list(set(b2))
    return b2


# returns all trigrams for the product with index idx
# remove and reduce trigrams by checking punctuation and stopwords
# dataf = df_allreview

def trigram(idx, dataf):    
    allrev = dataf.loc[idx]['all_reviews']  
    u = text_preprocess(allrev)   
    t1 = set(nltk.ngrams(u, 3))  
    t1 = list(t1)
    
    t2 = []

    for t in t1:
        if (t[0] in stop_words) or (t[1] in stop_words) or (t[2] in stop_words) or (t[0] in punc) or (t[1] in punc) or (t[2] in punc):
            continue

        if (t[0] not in punc) and (t[1] not in punc) and (t[2] not in punc) and (t[0] not in stop_words) and (t[1] not in stop_words) and (t[2] not in stop_words):
            t2.append(t)    
            
    t2 = list(set(t2))
    return t2

####
# POS-tag each candidates phrase and select those satisfying certain POS tag patterns
####

def tagging(tokens):
    tagged_tokens = nltk.pos_tag(tokens)
    return tagged_tokens


# For bigrams, selects and returns the list of final candidates

def candidate_pos(tokens, n):
    candidates = []

    # Popular phone brands - for bigrams, include it as a candidate if the first word is a phone brand 
    # (because second word likely to be phone model)
    brands = ('nokia','motorola','iphone','samsung','xiaomi','huawei',
              'siemens','sony','sonyericsson', 'ericsson',
              'palm','blackberry','htc','alcatel','benq','at&t','galaxy',
              'apple','asus','casio','google','kyocera','nec','sony','android')
    
    # JJR - adj comparative, JJS - adj superlative, 
    # RBR - adverb comparative, RBS - adverb superlative
    # CD - cardinal number
    unigram_tags = ('NN','NNS','NNP','NNPS')
    noun_tags = ('NN','NNS','NNP','NNPS')
    adjective_tags = ('JJ','JJR','JJS','CD')
    adverb_tags = ('RB','RBR','RBS')  # RB for 'not' and 'very'
    
    #verb_tags = ('VB','VBD','VBP','VBZ')
    
    if n == 1:                           # for unigrams
        tagged_tokens = tagging(tokens)
                   
        for t in tagged_tokens:         
            if t[0] in brands:
                candidates.append(t[0]) 
            
            if t[1] in unigram_tags:
                candidates.append(t[0])

    if n == 2:                          # for bigrams
        for x in tokens:
            t = tagging(x)
            
            if x[0] in brands:
                candidates.append(x) 
            
            if (t[0][1] in noun_tags) and (t[1][1] in noun_tags):
                candidates.append(x)                
            if (t[0][1] in adjective_tags) and (t[1][1] in noun_tags):
                candidates.append(x)
            if (t[0][1] in adverb_tags) and (t[1][1] in adjective_tags):
                candidates.append(x)

    if n == 3:                          # for trigrams
        for x in tokens:
            t = tagging(x)
            
            if x[0] in brands or x[1] in brands:
                candidates.append(x) 
            
            if (t[0][1] in noun_tags) and (t[1][1] in noun_tags) and (t[2][1] in noun_tags):
                candidates.append(x)                
            if (t[0][1] in adjective_tags) and (t[1][1] in noun_tags) and (t[2][1] in noun_tags):
                candidates.append(x)                
            if (t[0][1] in adverb_tags) and (t[1][1] in adjective_tags) and (t[2][1] in noun_tags):
                candidates.append(x)                
                    
    candidates = list(set(candidates))
    return candidates

####
# If using tf-idf for scoring, calculate tf-idf score for each candidate phrase
####

# Returns a dictionary {word : tfidf} for all words (unigrams) of the product
# create 3-element tuple so that it can be combined with bigram tuple for ranking

def unigram_tfidf(tokens, tfidf):
    u_tfidf = []
    
    for u in tokens:
        tup = (u, tfidf[u])   # Create tuple   
        u_tfidf.append(tup)
    
    return u_tfidf


# Returns a dictionary {bigram : tfidf} for all bigrams) of the product

def bigram_tfidf(tokens, tfidf):
    b_tfidf = []
    
    for b in tokens:
        b = list(b)
        tup = (b[0], b[1], tfidf[b[0]] + tfidf[b[1]]) # Create tuple 
        b_tfidf.append(tup)    
      
    return b_tfidf    


# Returns a dictionary {bigram : tfidf} for all bigrams) of the product

def trigram_tfidf(tokens, tfidf):
    t_tfidf = []
    
    for t in tokens:
        t = list(t)
        tup = (t[0], t[1], t[2], tfidf[t[0]] + tfidf[t[1]] + tfidf[t[2]]) # Create tuple 
        t_tfidf.append(tup)    
      
    return t_tfidf    

####
# If using tf for scoring, calculate tf score for each candidate phrase
####

def unigram_tf(tokens, tf):
    u_tf = []
    
    for u in tokens:
        tup = (u, '', '', tf[u])   # Create tuple   
        u_tf.append(tup)
            
    return u_tf


def bigram_tf(tokens, tf):
    b_tf = []
    
    for b in tokens:
        b = list(b)
        tup = (b[0], b[1], '', tf[b[0]] + tf[b[1]]) # Create tuple 
        b_tf.append(tup)    
      
    return b_tf    


def trigram_tf(tokens, tf):
    t_tf = []
    
    for t in tokens:
        t = list(t)
        tup = (t[0], t[1], t[2], tf[t[0]] + tf[t[1]] + tf[t[2]]) # Create tuple 
        t_tf.append(tup)    
      
    return t_tf   

#### 
# Find the review frequency of each candidate phrase (number of reviews of the product containing the phrase)
####

# returns a list [s, s, ...] if there are matches of s in data. If no match, returns []

def string_match(s, data):
    data = str(data)
    match = re.findall(re.escape(s), data.lower())
    
    return match


# dataf = df_review

def unigram_rf(tokens, idx, dataf,RF_WEIGHT):
    data = dataf.loc[idx]    
    n = data[1]                    # num_reviews
    
    u_finalscore = []
    u_rf = defaultdict(lambda: 0)  # review freq
    
    for u in tokens:
        s = u[0] 
        
        for i in range(2,n+2):
            match = string_match(s, data[i])
            if len(match) != 0:
                u_rf[u] += 1
        
        if u not in u_rf:
            score = 0
        else:                
            score = RF_WEIGHT * math.log10(u_rf[u]) + (u_rf[u]/n)   # give recurring unigrams more weight by * 2
        
        finalscore = score + u[3]        
        tup = (u[0], '', '', finalscore) 
        u_finalscore.append(tup)        
                
    return u_finalscore    


# dataf = df_review

def bigram_rf(tokens, idx, dataf,RF_WEIGHT):
    data = dataf.loc[idx]    
    n = data[1]                    # num_reviews
    
    b_finalscore = []
    b_rf = defaultdict(lambda: 0)  # review freq
    
    for b in tokens:
        s = b[0] + ' ' + b[1]
        
        for i in range(2,n+2):
            match = string_match(s, data[i])
            if len(match) != 0:
                b_rf[b] += 1  
                
        if b not in b_rf:
            score = 0
        else:                
            score = RF_WEIGHT * math.log10(b_rf[b]) + (b_rf[b]/n)

        finalscore = score + b[3]
        tup = (b[0], b[1], '', finalscore) 
        b_finalscore.append(tup)
        
    return b_finalscore  

# dataf = df_review

def trigram_rf(tokens, idx, dataf):
    data = dataf.loc[idx]    
    n = data[1]                    # num_reviews
    
    t_finalscore = []
    t_rf = defaultdict(lambda: 0)  # review freq
    
    for t in tokens:
        s = t[0] + ' ' + t[1] + ' ' + t[2]
        
        for i in range(2,n+2):
            match = string_match(s, data[i])
            if len(match) != 0:
                t_rf[t] += 1  
                
        if t not in t_rf:
            score = 0
        else:                
            score = math.log10(t_rf[t]) + (t_rf[t]/n)

        finalscore = score + t[3]
        tup = (t[0], t[1], t[2], finalscore) 
        t_finalscore.append(tup)
        
    return t_finalscore  

####
# Rank the final candidate phrases
####

# Returns number of words/phrases to output in summary

def num_words(idx, dataf,SUMMARY_SIZE_FACTOR):

    # get total number of words in all reviews (excluding stopwords)
    allrev = dataf.loc[idx]['all_reviews']  
    u = text_preprocess_clean(allrev)   
    n = len(u)
    numwords =  math.ceil(SUMMARY_SIZE_FACTOR * math.log10(n))  # set number of key phrases in summary
    return numwords


# Calculate the number of words/phrases in the summary based on the number of unique words in all reviews of the product
# Returns the summary words/phrases
# dataf = df_allreview

def rank_score(idx, utokens, btokens, ttokens, dataf,SUMMARY_SIZE_FACTOR):    
    candidates = []
    result = []
    
    summary_size = num_words(idx, dataf,SUMMARY_SIZE_FACTOR)
    
    # Concatenate the lists of unigrams and bigrams
    candidates = utokens + btokens + ttokens
    candidates = list(set(candidates))
    
    candidates_sorted = sorted(candidates, key=lambda tup: tup[3], reverse=True)    
    n = min(len(candidates_sorted), summary_size)
              
    for i in range(0,n):
        result.append(candidates_sorted[i])
    
    return result

####
# Calculate and check Levenshtein distance between each pair of candidate phrases
# Filter out very similar phrases and return the final summary phrases
####

# Levenshtein distance for each pair of candidate words (unigrams, bigrams, trigrams)
# Instead of stemming the words at the beginning, we use this method to remove words that are too similar 

def levenshtein(w1, w2):
    lev_dist = 1. - jellyfish.levenshtein_distance(w1, w2) / max(len(w1), len(w2))

    return lev_dist


# Checks Levenshtein distance between each pair of words/phrases
# If distance >= 0.8, reject the SHORTER word/phrase. If both words are the same length, do not reject any.
# Return a list of candidates as final results

def check_similarity(words, LEVENSHTEIN_THRESHOLD):
    tokens = []
    reject = []    
    result = {}
    length = len(words)

    # Convert tuples to strings
    for tup in words:
        s = tup[0] + ' ' + tup[1] + ' ' + tup[2]
        tokens.append(s.strip())
        
    # Check similarity between every pair of terms
    # If similarity >= 0.8, reject it as a candidate
    for i, w in enumerate(tokens):
        for j in range(0,length):
            if i != j and levenshtein(w, tokens[j]) >= LEVENSHTEIN_THRESHOLD:
                if len(w) < len(tokens[j]):
                    reject.append(w)
                    
                if len(w) > len(tokens[j]):
                    reject.append(tokens[j])

    # Remove rejected strings     
    result = list(set(tokens) - set(reject))
    
    return sorted(result)

####
# Main function for keyphrase summarizer
####

def main(index):
    
    if index >= 10429:
        print('Please enter a product id from 0 to 10428.')
        return
    
    # ADJUSTABLE PARAMETERS
    SUMMARY_SIZE_FACTOR = 5      # numwords =  math.ceil(SUMMARY_SIZE_FACTOR * math.log10(n)
    RF_WEIGHT = 2                # weight for unigrams and bigrams when calculating review frequency score
    LEVENSHTEIN_THRESHOLD = 0.8  # reject one candidate if threshold distance between two candidates >= 0.8  
    
    print('**** PRODUCT REVIEW SUMMARIZER ****\n')

    # STEP 1: search for the index and print data (index, product id, number of reviews, review text)
    num_reviews = search(index, df_review)
    
    # STEP 2: calculate tf-idf for product reviews' words
    tfreq, tf, tfidf = word_tfidf(index, idf, df_allreview)
    
    # STEP 3: tokenize and filter tokens to select candidates
    u1 = unigram(index, df_allreview)
    b1 = bigram(index, df_allreview)
    t1 = trigram(index, df_allreview)
 
    # STEP 4: filter candidates by POS tags    
    u2 = candidate_pos(u1, 1)
    b2 = candidate_pos(b1, 2)
    t2 = candidate_pos(t1, 3)
    
    # STEP 5.1: Use TF-IDF to calculate the score for each candidate
    u3 = unigram_tfidf(u2, tfidf)
    b3 = bigram_tfidf(b2, tfidf)  
    t3 = trigram_tfidf(t2, tfidf)  
        
    # STEP 5.2: Use TF to calculate the score for each candidate
    u4 = unigram_tf(u2, tf)
    b4 = bigram_tf(b2, tf)  
    t4 = trigram_tf(t2, tf)  
    
    # STEP 6: Count the number of reviews of the product each candidate is in. 
    u5 = unigram_rf(u4, index, df_review, RF_WEIGHT)
    b5 = bigram_rf(b4, index, df_review, RF_WEIGHT)
    t5 = trigram_rf(t4, index, df_review)
        
    # STEP 7: Select top-ranked candidates by final score and rank the candidates
    result = rank_score(index, u5, b5, t5, df_allreview, SUMMARY_SIZE_FACTOR)      
                      
    # STEP 8: Check Levenshtein distance to filter out very similar words (such as singular and plural forms). 
    summary = check_similarity(result, LEVENSHTEIN_THRESHOLD)
    
    print('\nSUMMARY KEYPHRASES:\n')
    for s in summary:
        print(s)

####
# CREATE DATAFRAMES AND COMPUTE idf FIRST
####

DATA_DIR = "data/"
file1 = os.path.join(DATA_DIR, 'asin_numreviews_review.csv')
file2 = os.path.join(DATA_DIR, 'asin_numreviews_allreview.csv')

df_review = utils.csv_to_dataframe(file1)
df_allreview = utils.csv_to_dataframe(file2)

idf = compute_idf(df_allreview)


####
# Enter the index of product (0 to 10428) to retrieve summary keyphrases.
####

product_index = 1005
main(product_index)
