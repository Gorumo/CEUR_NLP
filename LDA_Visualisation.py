import psycopg2
import re
import nltk
import pyLDAvis.gensim
import pandas as pd
import pickle as pk
from scipy import sparse as sp
from bs4 import BeautifulSoup
from autocorrect import spell
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from itertools import product
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from nltk.corpus import wordnet as wn
from autocorrect import spell
from collections import Iterable
from nltk import pos_tag, word_tokenize
from nltk.chunk.util import *
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

nltk.download('stopwords')
stopwords = stopwords.words('english')
lemmatizer2 = nltk.WordNetLemmatizer()
def lemmatizator(text):

    # Used when tokenizing words
    sentence_re = r'''(?x)      # 
            (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
          | \w+(?:-\w+)*        # words with optional internal hyphens
          | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
          | \.\.\.              # ellipsis
          | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
        '''

    # Taken from Su Nam Kim Paper.
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NNS|NN>} 
            {<NNP>*<NNP>}
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  
    """
    chunker = nltk.RegexpParser(grammar)
    #print(chunker)
    toks = nltk.regexp_tokenize(text, sentence_re)
    #print(toks)
    toks_correction = [x if x[0].isupper() or len(x) < 2 else spell(x) for x in toks]
    #print(toks_correction)
    postoks = nltk.tag.pos_tag(toks_correction)
    #print(postoks)
    tree = chunker.parse(postoks)
    #print(tree)
    terms = get_terms(tree)
    concepts = []
    for term in terms:
        if term and len(term) < 4:
            sentence = '_'.join(term)
            concepts.append(sentence)
    c = Counter(concepts)
    concepts = []
    for letter, count in c.most_common(len(c) // 3):
        concepts.append(letter)
    #print(concepts)
    return concepts

def get_continuous_chunks(model, text, label):
    # print(chunker_model.parse(pos_tag(word_tokenize(text))))
    # chunked = ne_chunk(pos_tag(word_tokenize(text)))
    chunked = model.parse(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree and subtree.label() == label:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk

def hyperym(x):
    #x = x.replace("_", " ")
    dog = ''
    group_ID = []
    if len(wn.synsets(x)):
        dog = wn.synsets(x)[0]
        group = dog.hypernyms()
        if group:
            group_ID.append(group[0].name().partition('.')[0])
    elif '_' in x:
        words = x.split('_')
        for s in words:
            try:
                dog = wn.synsets(s)[0]
                if dog.name().partition('.')[2][:1] == 'n':
                    group = dog.hypernyms()
                    if group:
                        group_ID.append(group[0].name().partition('.')[0])
            except:
                continue
    return group_ID

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        yield subtree.leaves()


def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    #word = stemmer.stem(word)
    word = lemmatizer2.lemmatize(word)
    return word


def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(3 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted


def get_terms(tree):
    for leaf in leaves(tree):
        term = [normalise(w) for w, t in leaf if acceptable_word(w)]
        yield term


def clear_data(htmltext):
    mystr = BeautifulSoup(htmltext, "html.parser")
    mystr = mystr.get_text()
    mystr = re.sub("^\s+|\n|\r|\s+$", ' ', mystr)
    mystr = mystr.replace('"', '')
    mystr = mystr.replace('"', '')
    return mystr


def get_doc_topic_dist(model, corpus, kwords=False):
    '''
    LDA transformation, for each doc only returns topics with non-zero weight
    This function makes a matrix transformation of docs in the topic space.
    '''
    top_dist = []
    keys = []

    for d in corpus:
        tmp = {i: 0 for i in range(num_topics)}
        tmp.update(dict(model[d]))
        vals = list(OrderedDict(tmp).values())
        top_dist += [array(vals)]
        if kwords:
            keys += [array(vals).argmax()]

    return array(top_dist), keys


def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isdigit()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 3] for doc in docs]

    # Lemmatize all words in documents.
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    return docs

NamedIndividual = '<http://www.semanticweb.org/Urban#%s> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#NamedIndividual> .'
CustomType = '<http://www.semanticweb.org/Urban#%s> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.semanticweb.org/Urban#%s> .'
Description = '<http://www.semanticweb.org/Urban#%s> <http://www.semanticweb.org/Urban#%s> "%s" .'
Relation = '<http://www.semanticweb.org/Urban#%s> <http://www.semanticweb.org/Urban#%s> <http://www.semanticweb.org/Urban#%s> .'
Rating = '<http://www.semanticweb.org/Urban#%s> <http://www.semanticweb.org/Urban#rating> "%f"^^<http://www.w3.org/2001/XMLSchema#integer> .'
conn = psycopg2.connect("dbname=quakitnlp user=romanov")
cursor = conn.cursor()
cursor.execute("SELECT id,author_id,description FROM current_scenario WHERE length(description)>50;")
rows = cursor.fetchall()


ALL_CONCEPTS = []
ALL_SUB = []
# criteria = ['visibility','centrality','connectivity','accessibility','density','distribution','connectivity', 'walkability']


from gensim.corpora import Dictionary

for row in rows:
    sub_id, author_id, description = row
    description = clear_data(description)
    ALL_SUB.append(description)
    keywords = lemmatizator(description)
    for concept_ID in keywords:
        ALL_CONCEPTS.append(concept_ID)


document = docs_preprocessor(ALL_SUB)

docs = docs_preprocessor(ALL_CONCEPTS)
# Add bigrams and trigrams to docs (only ones that appear 2 times or more).
bigram = Phrases(docs, min_count=3)
trigram = Phrases(bigram[docs])

for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)
    for token in trigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

print('Number of unique words in initital documents:', len(dictionary))

# Filter out words that occur less than 10 documents, or more than 20% of the documents.
dictionary.filter_extremes(no_below=3, no_above=0.2)
print('Number of unique words after removing rare and common words:', len(dictionary))

corpus = [dictionary.doc2bow(doc) for doc in document]

print(dictionary)
print('Number of unique tokens: %d' % len(dictionary))
print(corpus)
print('Number of documents: %d' % len(corpus))

# Set training parameters.
num_topics = 100
chunksize = 400 # size of the doc looked at every pass
passes = 20 # number of passes through documents
iterations = 400
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                       alpha='auto', eta='auto', \
                       iterations=iterations, num_topics=num_topics, \
                       passes=passes, eval_every=eval_every)

#pyLDAvis.enable_notebook()



visualisation = pyLDAvis.gensim.prepare(model, corpus, dictionary)

pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

