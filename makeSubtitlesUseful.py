#!/usr/bin/env python
#These python classes allow for easy parsing of turns and text from the TED subtitle files. 
import os
import re
import codecs
import json
import glob
import datetime
from bs4 import BeautifulSoup as Soup
import warnings
import nltk
from nltk.corpus import stopwords
import string
from gensim import corpora

SUBTITLE_DIR = '/home/sidd/Documents/Research/TEDProject/subtitles' 
OUT_PREFIX = 'htmm/data/'
TURN_GROUP_SIZE = 15

def tedx_xml_reader(filename):
    contents = ensure_unicode(codecs.open(filename, 'r', 'utf8').read())
    soup = Soup(contents)
    trans = soup.transcript
    if not trans:
        warnings.warn('File %s has no XML transcript element.' % filename, RuntimeWarning)
    else:
        for turn_elem in trans.find_all('text'):
            t ={}
            t['start'] = process_tedx_duration(turn_elem['start'])
            t['duration'] = None
            t['finish'] = None            
            if turn_elem.has_attr('dur'):
                t['duration'] = process_tedx_duration(turn_elem['dur'])
                t['finish'] = t['start'] + t['duration']
            t['text'] = unicode(turn_elem.string)                            
            yield Turn(t)


def process_tedx_duration(s):
    val = s.split('.')
    secs = int(val[0])
    microsecs = 0   
    if len(val) > 1:
        microsecs = int(val[1])
    return datetime.timedelta(seconds=secs, microseconds=microsecs)


def ted_txt_reader(filename):
    contents = ensure_unicode(codecs.open(filename, 'r', 'utf8').read().strip())
    turn_re = re.compile(r"\n\s*\n", re.M)
    turns = turn_re.split(contents)
    for turn in turns:
        turn = turn.splitlines()
        start, finish = re.split(r'\s*-->\s*', turn[1])
        start = process_ted_duration(start)
        finish = process_ted_duration(finish)        
        duration = finish - start
        text = " ".join([line.strip() for line in turn[2: ]])
        t = {'start': start, 'duration': duration, 'finish': finish, 'text': text}
        yield Turn(t)


def process_ted_duration(s):
    hr, minute, secs = s.split(':')
    secs, microsecs = secs.split(',')
    return datetime.timedelta(hours=int(hr), minutes=int(minute), seconds=int(secs), microseconds=int(microsecs))
    

def ensure_unicode(s):
    try:
        s = unicode(s)
    except UnicodeDecodeError:
        s = str(s).encode('string_escape')
        s = unicode(s)
    return s    


#TODO: Write get_clean_words equivalent for turns.  Use this to get both the dict and the HTMM-formatted data.
class Transcript:
    def __init__(self, reader=None, filename=None):
        print filename 
        self.filename = filename
        self.turns = list(reader(filename))

    def to_json(self):
        return [t.to_json() for t in self.turns]
        
    def __len__(self):
        return len(self.turns)
    
    def __unicode__(self):
        return "\n".join([str(t) for t in self.turns])

    #Gets one line with all desired words for LDA
    def get_all_words(self, tag_regex = None):
        return ' '.join([t.get_pos_words(tag_regex) for t in self.turns])

    def get_clean_words(self, tag_regex = None):
        temp_list = [cur_turn.get_clean_pos_words(tag_regex) for cur_turn in self.turns]
        all_words = [innerelem for turn_list in temp_list for innerelem in turn_list]
        return all_words  
        #start = self.get_all_words(tag_regex = tag_regex)
        #temp = [word.encode('ascii','ignore').lower().translate(string.maketrans("",""), string.punctuation) for word in start.split(' ')]
        #return  [word for word in temp if word != '' and word not in stopwords.words('english')] 
         

#TODO: See if get_clean_pos_words works... 
class Turn:
    def __init__(self, t):
        # All text element seem to have start:        
        self.start = t['start']
        # All elements have text (sometimes empty string):
        self.text = t['text']
        # Not all such elements have specified durations:
        self.duration = t['duration']
        self.finish = t['finish']

    def pos(self, tag_regex=None):
        toks = nltk.word_tokenize(self.text)
        lems = nltk.pos_tag(toks)
        if tag_regex:
            lems = [lem for lem in lems if tag_regex.search(lem[1])]
        return lems

    def to_json(self):
        return {'duration': self.duration, 'start': self.start, 'finish': self.finish, 'text': self.text}

    def __unicode__(self):
        return u'%s\t%s' % (self.start, self.text)

    def get_pos_words(self, tag_regex = None):
        return ' '.join([lem[0] for lem in self.pos(tag_regex = tag_regex)])
   
    def get_clean_pos_words(self, tag_regex = None):
        words = [lem[0] for lem in self.pos(tag_regex = tag_regex)]
        temp = [word.encode('ascii','ignore').lower().translate(string.maketrans("",""), string.punctuation) for word in words]
        return  [word for word in temp if word != '' and word not in stopwords.words('english')] 
 
def get_one_doc_per_line(out_loc = 'sample_out.txt', filenames = None):
    #my_regex = re.compile(r"^(NN([^P]|$)|V|JJ|RB)", re.I)
    my_regex = re.compile(r"^(JJ|RB)", re.I)
    #open(out_loc,'w').write('\n'.join([Transcript(filename = SUBTITLE_DIR + '/' + cur_file, reader = ted_txt_reader).get_all_words(tag_regex = my_regex) for cur_file in os.listdir(SUBTITLE_DIR)]))
    #writable = '\n'.join([Transcript(filename = SUBTITLE_DIR + '/' + cur_file, reader = ted_txt_reader).get_all_words(tag_regex = my_regex) for cur_file in os.listdir(SUBTITLE_DIR)])
    writable = '\n'.join([Transcript(filename = cur_file, reader = ted_txt_reader).get_all_words(tag_regex = my_regex) for cur_file in filenames])
    writer = open(out_loc, 'w')
    writer.write(writable.encode('ascii', 'ignore')) 

#TODO: Write turn by turn functions

#Gets clean words from htmm
def clean_htmm_words(doc_line):
    temp = [word.encode('ascii','ignore').lower().translate(string.maketrans("",""), string.punctuation) for word in doc_line.split(' ')]
    return  [word for word in temp if word != '' and word not in stopwords.words('english')] 

#Gets Dictionary
def get_htmm_dict(filenames = None, tag_regex = None):
    print "Getting dictionary..."
    #words = [clean_htmm_words(Transcript(filename = cur_file, reader = ted_txt_reader).get_all_words() for cur_file in filenames)]
    words = [Transcript(filename = cur_file, reader = ted_txt_reader).get_clean_words(tag_regex = tag_regex) for cur_file in filenames]
    return corpora.Dictionary(words)

#Gets the desired representation of a single sentence/chunk for HTMM
def get_htmm_line(list_of_words, word_dict):
    cur_len = len(list_of_words) 
    cur_line = str(cur_len) + ' ' + ' '.join([str(word_dict.token2id[cur_word]) for cur_word in list_of_words]) + '\n'
    return cur_line 

#Gets the location of where to write the HTMM data
def get_write_filename(filename):
    return OUT_PREFIX + filename.split('/')[-1].split('.')[0] + '.txt' 

#Gets the HTMM-formatted data given a generator of filenames
def get_htmm_data(filenames = None, word_dict = None, tag_regex = None):
    file_list = open('htmm/file_list.txt', 'w') 
    print "Now getting HTMM data..."
    for cur_file in filenames:
        file_list.write(get_write_filename(cur_file) + '\n')
        with open(get_write_filename(cur_file), 'w') as writer:
            cur_transcript = Transcript(filename = cur_file, reader = ted_txt_reader)
            for cur_turn in cur_transcript.turns:
                cur_line = get_htmm_line(cur_turn.get_clean_pos_words(tag_regex = tag_regex), word_dict)
                writer.write(cur_line)
                #print cur_line
    file_list.close()

#Gets a turn as a document (a line of text for the document) 
def get_turn_as_doc(turn, tag_regex = None):
    return ' '.join(turn.get_clean_pos_words(tag_regex = tag_regex)) + '\n'

#Converts a group of turns into a single doc_line
def get_turns_as_doc(turns, tag_regex = None):
    temp = [turn.get_clean_pos_words(tag_regex = tag_regex) for turn in turns]
    return ' '.join([' '.join(item) for item in temp if len(item) > 0]) + '\n'
    #return ' '.join([' '.join(turn.get_clean_pos_words(tag_regex = tag_regex)) for turn in turns]) + '\n'

#Gets doc-per-line for LDA treating each group of TURN_GROUP_SIZE turns as a separate document and only including words found by tag_regex
def get_turn_docs(filenames = None, tag_regex = None):
    out_writer = open('turnsAsDocs/data/test_adjective_lines.txt', 'w')
    file_list_writer = open('turnsAsDocs/data/adjective_file_list.txt', 'w')
    for cur_file in filenames:
        cur_transcript = Transcript(filename = cur_file, reader = ted_txt_reader)
        cur_index = 0
        while cur_index < len(cur_transcript.turns): 
            cur_turns = cur_transcript.turns[cur_index:min(cur_index + TURN_GROUP_SIZE, len(cur_transcript.turns))] 
            cur_line = get_turns_as_doc(cur_turns,tag_regex = tag_regex) 
            out_writer.write(cur_line)
            cur_index = cur_index + TURN_GROUP_SIZE
            file_list_writer.write(cur_file + '\n') 
    out_writer.close()

if __name__ == '__main__':
    def demo(filenames=None, reader=None):
        # Stream through all of the TEDx transcripts, print the text of their dialogues and
        # sum up all the transcript lengths using the turn durations:
        dur = datetime.timedelta()
        for filename in filenames:
            print filename
            trans = Transcript(filename=filename, reader=reader)
            for t in trans.turns:
                # Look non-proper nouns, verbs, adjectives, and adverbs:
                print t.pos(tag_regex=re.compile(r"^(NN([^P]|$)|V|JJ|RB)", re.I))
                if t.duration != None:
                    dur += t.duration
        print dur

    def tedx_demo():
        filenames = glob.iglob(os.path.join('..', 'tedx_transcripts', '*.xml'))
        demo(filenames=filenames, reader=tedx_xml_reader)

    def ted_demo():
        filenames = glob.iglob(os.path.join('/home', 'sidd', 'Documents', 'Research', 'TEDProject', 'small_subs', '*.srt'))
        demo(filenames=filenames, reader=ted_txt_reader)
    
    filenames = glob.iglob(os.path.join('/home', 'sidd', 'Documents', 'Research', 'TEDProject', 'small_subs', '*.srt'))
    my_regex = re.compile(r"^(JJ|RB)", re.I)
    get_turn_docs(filenames, tag_regex = my_regex)    

'''
    filenames = glob.iglob(os.path.join('/home', 'sidd', 'Documents', 'Research', 'TEDProject', 'small_subs', '*.srt'))
    token_dict = get_htmm_dict(filenames)
    filenames = glob.iglob(os.path.join('/home', 'sidd', 'Documents', 'Research', 'TEDProject', 'small_subs', '*.srt'))
    get_htmm_data(filenames = filenames, word_dict = token_dict)
    with open(OUT_PREFIX + 'numwords.txt', 'w') as num_words_writer:
        num_words_writer.write(str(len(token_dict.token2id.keys())))
''' 
    #get_one_doc_per_line(out_loc = 'data/adjectives_adverbs_ted_corpus.txt', filenames = filenames)
