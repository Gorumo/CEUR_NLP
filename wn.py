from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn

words = ['house', 'resident', 'area', 'road', 'community development space', 'open area', 'community', 'easy access', 'safety', 'health', 'comfort', 'design idea', 'western portion']
for x in words:
    print(x)
    dog = ''
    if len(wn.synsets(x)):
        dog = wn.synsets(x)[0]
        group = dog.hypernyms()
        print(dog, group[0].name().partition('.')[0])
    elif ' ' in x:
        words = x.split(' ')
        print(words)
        for s in words:
            try:
                dog = wn.synsets(s)[0]
                if dog.name().partition('.')[2][:1] == 'n':
                    group = dog.hypernyms()
                    print(dog, group[0].name().partition('.')[0])
            except:
                continue


#print(dog.hyponyms())  # doctest: +ELLIPSIS
#print(dog.member_holonyms())
#print(dog.root_hypernyms())
#print(wn.synset('patio.n.01').lowest_common_hypernyms(wn.synset('courtyard.n.01')))
