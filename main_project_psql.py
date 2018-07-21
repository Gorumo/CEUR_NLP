import psycopg2
import string
import os
from bs4 import BeautifulSoup
import uuid
import nltk
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


def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """

    # init the stemmer
    stemmer = SnowballStemmer('english')

    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'),
                                                                                    ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])

    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase

    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase

    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase

    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,

        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,

        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,

        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,

        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,

        'prev-iob': previob,

        'contains-dash': contains_dash,
        'contains-dot': contains_dot,

        'all-caps': allcaps,
        'capitalized': capitalized,

        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,

        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }


def to_conll_iob(annotated_sentence):
    """
    `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token

        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))
    return proper_iob_tokens


def read_gmb(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]

                        standard_form_tokens = []

                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]

                            if ner != 'O':
                                ner = ner.split('-')[0]

                            if tag in ('LQU', 'RQU'):  # Make it NLTK compatible
                                tag = "``"

                            standard_form_tokens.append((word, tag, ner))

                        conll_tokens = to_conll_iob(standard_form_tokens)

                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield [((w, t), iob) for w, t, iob in conll_tokens]


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
    word = lemmatizer.lemmatize(word)
    return word


def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(3 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted


def get_terms(tree):
    for leaf in leaves(tree):
        for w, t in leaf:
            if t == "JJ":
                jj.write(w+'\n')
        term = [normalise(w) for w, t in leaf if acceptable_word(w)]
        yield term


def clear_data(htmltext):
    mystr = BeautifulSoup(htmltext, "html.parser")
    mystr = mystr.get_text()
    mystr = re.sub("^\s+|\n|\r|\s+$", ' ', mystr)
    mystr = mystr.replace('"', '')
    mystr = mystr.replace('"', '')
    return mystr


def similar(all_words):
    all_words = sorted(all_words)
    search_results = sorted(all_words, key=len)
    CONCEPTS = []
    while search_results:
        super_concepts = []
        some_word = search_results.pop(0)
        super_concepts.append(some_word)
        for x in search_results:
            #ratio = difflib.SequenceMatcher(None, some_word, x).ratio()
            #if ratio > 0.9:
            #    super_concepts.append(x)
            try:
                word1 = wordnet.synsets(some_word)
                #print(word1[0].name())
                word2 = wordnet.synsets(x)
                for sense1, sense2 in product(word1, word2):
                    d = wordnet.wup_similarity(sense1, sense2)
                    if d > 0.7:
                        super_concepts.append(x)
            except:
                continue
        CONCEPTS.append(super_concepts)
        for x in super_concepts:
            if x in search_results:
                search_results.remove(x)

    return sorted(CONCEPTS)


class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)

        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)

nltk.download('maxent_ne_chunker')
nltk.download('words')

nltk.download('stopwords')
stopwords = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()
stemmer = PorterStemmer()

corpus_root = "gmb-2.2.0"  # Make sure you set the proper path to the unzipped corpus

with open("quakit_base_clear.owl") as file:
    ALL_DATA = [row.strip() for row in file]

jj = open('jj.txt', 'w')
geo = open('geo.txt', 'w')

NamedIndividual = '<http://www.semanticweb.org/Urban#%s> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#NamedIndividual> .'
CustomType = '<http://www.semanticweb.org/Urban#%s> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.semanticweb.org/Urban#%s> .'
Description = '<http://www.semanticweb.org/Urban#%s> <http://www.semanticweb.org/Urban#%s> "%s" .'
Relation = '<http://www.semanticweb.org/Urban#%s> <http://www.semanticweb.org/Urban#%s> <http://www.semanticweb.org/Urban#%s> .'
Rating = '<http://www.semanticweb.org/Urban#%s> <http://www.semanticweb.org/Urban#rating> "%f"^^<http://www.w3.org/2001/XMLSchema#integer> .'
conn = psycopg2.connect("dbname=quakitnlp user=romanov")
cursor = conn.cursor()
cursor.execute("SELECT id,author_id,description FROM current_scenario WHERE length(description)>50;")
rows = cursor.fetchall()
# Each word has at least 5 characters
# Each phrase has at most 3 words
# Each keyword appears in the text at least 4 times
ALL_CONCEPTS = []

reader = read_gmb(corpus_root)
data = list(reader)
training_samples = data[:int(len(data) * 0.9)]
test_samples = data[int(len(data) * 0.9):]

#print("#training samples = %s" % len(training_samples))
#print("#test samples = %s" % len(test_samples))

chunker_model = NamedEntityChunker(training_samples)

for row in rows:
    sub_id, author_id, description = row
    keywords = lemmatizator(description)
    submission_ID = 'sub_'+str(sub_id)
    ALL_DATA.append(NamedIndividual % submission_ID)
    ALL_DATA.append(CustomType % (submission_ID, 'Submission'))
    ALL_DATA.append(Description % (submission_ID, 'submissionDesciption', clear_data(description)))
    # geo
    # z = get_continuous_chunks(chunker_model, description, 'geo')
    #for w in z:
    #    if any(x.isupper() for x in w):
    #        geo.write(w + '\n')
    for concept_ID in keywords:
        ## для статистики
        if 'visibility' in concept_ID or 'visibility' == concept_ID:
            criterion_id = 1
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
        ## потом удалить
        # concept_ID = concept_ID.replace(" ", "_")
        ALL_DATA.append(Relation % (submission_ID, 'hasConcept', concept_ID))
        if concept_ID not in ALL_CONCEPTS:
            ALL_CONCEPTS.append(concept_ID)
            ALL_DATA.append(NamedIndividual % concept_ID)
            ALL_DATA.append(CustomType % (concept_ID, 'Concept'))
            group_ID = hyperym(concept_ID)
            if group_ID:
                for x in group_ID:
                    x = 'gr_'+x
                    ALL_DATA.append(NamedIndividual % x)
                    ALL_DATA.append(CustomType % (x, 'Group'))
                    ALL_DATA.append(Relation % (concept_ID, 'belongsToGroup', x))


cursor.execute("SELECT id,scenario_id,comment FROM review;")
rows = cursor.fetchall()
for row in rows:
    review_id, sub_id, description = row
    submission_ID = 'sub_'+str(sub_id)
    review_ID = 'rev_'+str(review_id)
    if description:
        ALL_DATA.append(Relation % (submission_ID, 'hasReview', review_ID))
        ALL_DATA.append(NamedIndividual % review_ID)
        ALL_DATA.append(CustomType % (review_ID, 'Review'))
        ALL_DATA.append(Description % (review_ID, 'reviewText', clear_data(description)))
    cursor.execute(
        "SELECT criterion_id, SUM(CASE WHEN positive = True THEN 1 ELSE 0 END), COUNT(positive) FROM review WHERE scenario_id = %d GROUP BY criterion_id;" % sub_id)
    criterion_rows = cursor.fetchall()
    for criterion in criterion_rows:
        criterion_id, yes, vote = criterion
        cursor.execute(
            "SELECT COUNT(worse_id) FROM vote WHERE worse_id = %d AND criterion_id = %d GROUP BY criterion_id;" % (
                sub_id, criterion_id))
        vote_rows = cursor.fetchone()
        if vote_rows:
            (result,) = vote_rows
            vote += result
        cursor.execute(
            "SELECT COUNT(better_id) FROM vote WHERE better_id = %d AND criterion_id = %d GROUP BY criterion_id;" % (
                sub_id, criterion_id))
        vote_rows = cursor.fetchone()
        if vote_rows:
            (result,) = vote_rows
            vote += result
            yes += result
        criterion_ID = uuid.uuid4()
        ALL_DATA.append(NamedIndividual % criterion_ID)
        ALL_DATA.append(Relation % (submission_ID, 'isRated', criterion_ID))
        ALL_DATA.append(CustomType % (criterion_ID, 'Criteria'))
        ALL_DATA.append(Rating % (criterion_ID, yes*100/vote))
        cursor.execute("SELECT name,description FROM criterion WHERE id = %d;" % criterion_id)
        criterion_info = cursor.fetchone()
        (criterion_name, criterion_description) = criterion_info
        ALL_DATA.append(Description % (criterion_ID, 'criteriaName', criterion_name))
        ALL_DATA.append(Description % (criterion_ID, 'criteriaDescription', clear_data(criterion_description)))


cursor.execute('SELECT id,name FROM "user";')
rows = cursor.fetchall()
for row in rows:
    user_id, name = row
    user_ID = 'user_'+str(user_id)
    cursor.execute("SELECT id FROM current_scenario WHERE author_id = %d;" % user_id)
    db_row = cursor.fetchone()
    if db_row:
        (sub_id,) = db_row
        submission_ID = 'sub_'+str(sub_id)
        ALL_DATA.append(NamedIndividual % user_ID)
        ALL_DATA.append(CustomType % (user_ID, 'User'))
        ALL_DATA.append(Description % (user_ID, 'name', name))
        ALL_DATA.append(Relation % (user_ID, 'hasSubmission', submission_ID))

with open('quakit_base.owl', 'w') as file:
    print(*ALL_DATA, file=file, sep='\n')

jj.close()
geo.close()

with open("jj.txt") as jj:
    data = [row.strip() for row in jj]
cntr = Counter(data)
#print(cntr)

#with open("geo.txt") as jj:
#    data = [row.strip() for row in jj]
#cntr = Counter(data)
#print(cntr)