import psycopg2
import re
import nltk
from bs4 import BeautifulSoup
from autocorrect import spell
from nltk.corpus import stopwords
from collections import Counter

nltk.download('stopwords')
stopwords = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()

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


def clear_data(htmltext):
    mystr = BeautifulSoup(htmltext, "html.parser")
    mystr = mystr.get_text()
    mystr = re.sub("^\s+|\n|\r|\s+$", ' ', mystr)
    mystr = mystr.replace('"', '')
    mystr = mystr.replace('"', '')
    return mystr


def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        yield subtree.leaves()


def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    #word = stemmer.stem(word)
    word = lemmatizer.lemmatize(word)
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


conn = psycopg2.connect("dbname=quakitnlp user=romanov")
cursor = conn.cursor()
cursor.execute("SELECT id,author_id,description FROM current_scenario WHERE length(description)>50;")
rows = cursor.fetchall()


criteria = ['visibility','centrality','connectivity','accessibility','density','distribution', 'accessibility','connectivity', 'walkability']
criteria_id = [1,2,3,4,581,581,582,582,946]


for row in rows:
    rows = cursor.fetchall()
    sub_id, author_id, description = row
    keywords = lemmatizator(description)
    for concept_ID in keywords:
        if 'walkability' in concept_ID:
            criterion_id = 946
            vote = 0
            yes = 0
            cursor.execute("SELECT COUNT(worse_id) FROM vote WHERE worse_id = %d AND criterion_id = %d GROUP BY criterion_id;" % (sub_id, criterion_id))
            vote_rows = cursor.fetchone()
            if vote_rows:
                (result,) = vote_rows
                vote += result
            cursor.execute("SELECT COUNT(better_id) FROM vote WHERE better_id = %d AND criterion_id = %d GROUP BY criterion_id;" % (sub_id, criterion_id))
            vote_rows = cursor.fetchone()
            if vote_rows:
                (result,) = vote_rows
                vote += result
                yes += result
            if vote:
                print(sub_id, criterion_id, yes*100/vote)