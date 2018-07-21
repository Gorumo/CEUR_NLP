import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
nltk.download('maxent_ne_chunker')
nltk.download('words')

sentence = "Njongo_Avenue"
print(ne_chunk(pos_tag(word_tokenize(sentence))))