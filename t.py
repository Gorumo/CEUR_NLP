import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag, ne_chunk
from autocorrect import spell
from nltk.corpus import wordnet as wn

nltk.download('maxent_ne_chunker')
nltk.download('words')

nltk.download('stopwords')
stopwords = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()
stemmer = PorterStemmer()

def lemmatizator(text):
    text = "The proposal bases in the idea of the different grids to provide the orientation for the houses creating public spaces on the inside for the community to share. Creating privacy inside and open to new public spaces. Leaving some areas outside as buffers between the community and the city that can be shared."
    # Used when tokenizing words
    sentence_re = r'''(?x)          # set flag to allow verbose regexps
            (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
          | \w+(?:-\w+)*        # words with optional internal hyphens
          | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
          | \.\.\.              # ellipsis
          | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
        '''



    # Taken from Su Nam Kim Paper...
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NNS|NN>}  # Nouns and Adjectives, terminated with Nouns
            {<NNP>*<NNP>}
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)
    #print(chunker)
    toks = nltk.regexp_tokenize(text, sentence_re)
    print(toks)
    toks_correction = [x if x[0].isupper() or len(x) < 2 else spell(x) for x in toks]
    print(toks_correction)
    postoks = nltk.tag.pos_tag(toks_correction)
    print(postoks)
    tree = chunker.parse(postoks)
    print(tree)
    terms = get_terms(tree)
    concepts = []
    for term in terms:
        sentence = ' '.join(term)
        print(sentence)
        # type, like GEO, ORG ...
        # print(ne_chunk(pos_tag(word_tokenize(sentence))))
        concepts.append(sentence)
    print(hyperym(concepts))
    print(concepts)
    return concepts


def hyperym(words):
    for x in words:
        try:
            dog = wn.synsets(x)[0]
            group = dog.hypernyms()
            print(group)
            print(x, " - is a part of: ", group[0].name().partition('.')[0])
        except:
            continue

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
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted


def get_terms(tree):
    for leaf in leaves(tree):
        term = [normalise(w) for w, t in leaf if acceptable_word(w)]
        yield term

print(lemmatizator(''))