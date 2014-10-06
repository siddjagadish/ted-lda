#This script converts a text file where each line is a separate document to David Blei's LDA-C format.  There are parameters to remove stopwords and to lemmatize words.  Additionally, MIN_NUM_DOCS specifies the minimum number of documents in which a word must appear to be included in the model, and MIN_NUM_PER_DOC specifies the minimum number of times a word must appear in a document for it to be included in the LDA-C representation of that document. 
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer  
from gensim import corpora, models, similarities
import string
import operator 
import os
from collections import Counter


SUBTITLE_DIR = '/home/sidd/Documents/Research/TEDProject/subtitles'
MIN_LEN = 5


LEMMATIZE = True
MIN_NUM_DOCS = 10 #Minimum number of documents a word must appear in to be included in LDA Model
MIN_NUM_PER_DOC = 1 #Minimum number of times a word must appear in a single document to appear in the LDA representation of that document


#Gets unique values in list
def f1(seq):
   # not order preserving
   set = {}
   map(set.__setitem__, seq, [])
   return set.keys()


#This function checks if a line is text, or marking time, or blank
def isEligible(line):
    if len(line) < MIN_LEN: return False
    if '-->' in line: return False
    return True

#Takes a single .srt TED transcript and turns it into a list of the constituent words 
def getSingleBowFromTED(filename, remove_stopwords = True, lemmatize = False):
    lem = WordNetLemmatizer()
    basiclist = ['(Laughter)', '(Applause)']
    lines = open(SUBTITLE_DIR + '/' + filename, 'r').readlines()
    lines = [line.strip('\n') for line in lines if isEligible(line)]
    bow = [line.split() for line in lines]
    bow = [word.lower().translate(string.maketrans("",""), string.punctuation) for line in bow for word in line if word not in basiclist]
    if remove_stopwords: 
        bow = [word for word in bow if word not in stopwords.words('english')]
    if lemmatize:
        bow = [lem.lemmatize(word) for word in bow] 
    return bow


#Returns list of lists.  Inner lists are of constituent words in a particular ted talk.  Outer list is over ted talks.
def getAllBowFromTED(remove_stopwords = True, lemmatize = False):
    return [getSingleBowFromTED(filename, remove_stopwords = remove_stopwords, lemmatize = lemmatize) for filename in os.listdir(SUBTITLE_DIR)] 


#Read in data from AP Corpus (for testing)
def getBow(file_loc):
    lines = open(file_loc, 'r').readlines()
    formatted_lines = [line for idx, line in enumerate(lines) if lines[idx - 1] == '<TEXT>\n']
    bow = [line.split() for line in formatted_lines]
    bow = [[word.lower().translate(string.maketrans("",""), string.punctuation) for word in line] for line in bow] #Gets rid of punctuation
    bow = [[word for word in line if word not in stopwords.words('english')] for line in bow]
    return bow

#Cleans up a line for LDA usage
def cleanDocFromLineDoc(line, remove_stopwords = True, lemmatize = False):
    words = line.strip('\n').split(' ')
    words = [word.lower().translate(string.maketrans("",""), string.punctuation) for word in words]  
    if remove_stopwords:
        words = [word for word in words if word not in stopwords.words('english') and word != '']
    return words
 
#Reads in data from a one doc per line file
def getBowFromLineDoc(file_loc):
    lines = open(file_loc, 'r').readlines()
    return [cleanDocFromLineDoc(line) for line in lines]    


#Gets vocabulary map (word:id)
def getVocabMap(bow):
    all_words = [word for doc in bow for word in doc]
    unique_words = f1(all_words)
    vocab_map = {word:idx for idx,word in enumerate(unique_words)}
    return vocab_map


#Processes a doc and turns it into word counts
def processDoc(doc):
    converted_doc = [vocab_map[word] for word in doc]
    temp_counter = Counter(converted_doc) 
    lda_formatted = [str(key) + ':' + str(val) for key,val in temp_counter.items()] 
    return lda_formatted


#Now take converted docs and remove words that only occur in one document
def getExcludeWordList(initial_docs, min_number_of_documents = 2):
    word_list = [word.split(':')[0] for doc in initial_docs for word in doc]
    word_counter = Counter(word_list)
    exclude_word_list = [int(key) for key,val in word_counter.items() if val < min_number_of_documents] 
    return exclude_word_list

 
#Now remove instances of one-document words
def removeExcludedWords(doc, exclude_word_list, min_num_occurrences_in_doc = 1):
    return [item for item in doc if int(item.split(':')[0]) not in exclude_word_list and int(item.split(':')[1]) >= min_num_occurrences_in_doc] 


#Converts ids of a doc to the new ones
def convertIds(doc, convert_ids_map):
    return [str(convert_ids_map[int(word.split(':')[0])]) + ':' + word.split(':')[1] for word in doc] 

print 'Getting Initial Stuff...'

#Get intial stuff
bow = getBowFromLineDoc('turnsAsDocs/data/doclines.txt')
vocab_map = getVocabMap(bow)
reverse_vocab_map = {val:key for key,val in vocab_map.items()}

print 'Converting to dense docs...'

#Convert initial docs to dense docs (not containing words that appear in documents insufficient number of times) 
initial_docs = [processDoc(doc) for doc in bow]
exclude_word_list = getExcludeWordList(initial_docs, min_number_of_documents = MIN_NUM_DOCS)
dense_docs = [removeExcludedWords(doc, exclude_word_list, min_num_occurrences_in_doc=MIN_NUM_PER_DOC) for doc in initial_docs]

print 'Converting old ids to new ids...'

#Now convert old ids to new ids (so range only covers 'dense' words)
remaining_words = f1([int(item.split(':')[0]) for doc in dense_docs for item in doc])
convert_ids_map = {word:idx for idx,word in enumerate(remaining_words)} 
final_dense_docs = [convertIds(doc, convert_ids_map) for doc in dense_docs]
reverse_final_vocab = {convert_ids_map[word]:reverse_vocab_map[word] for word in remaining_words}

print 'Writing output...'

writable_docs = '\n'.join([str(len(doc)) + ' ' + (' ').join(doc) for doc in final_dense_docs])
#NOTE:Relying on reverse_final_vocab already being sorted
writable_vocab = '\n'.join(reverse_final_vocab.values())

with open('turnsAsDocs/data/five_turns_corpus.dat', 'w') as writer:
    writer.write(writable_docs)

with open('turnsAsDocs/data/five_turns_vocab.txt', 'w') as vocab_writer:
    vocab_writer.write(writable_vocab) 


