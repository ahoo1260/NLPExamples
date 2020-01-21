
import re
from nltk.corpus import stopwords
from stemming.porter2 import stem


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
punct = "/-',#$%()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&-[]'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi' ,'1':'','2':'','3':'','4':'','5':'','6':'','7':'','8':'','9':'',"0":''}
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
#Please comment bad_words if you don't have any
# bad_words_dementia=['mhm', 'exc', 'uh', 'gram', 'mm', '...', 'uk', "xxx", "laughs", "throat", "um","okay","thats","shes","action","na"]
bad_words=["monday","sunday","saturday","tuesday","thursday","friday","wednesday","time","year","day","month","said",
           "percent","years","na","mon","june","july","aug","augest","may","march","sept","week","report"]
forbidden_words = set(stopwords.words('english')).union(set(bad_words))

def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")




def stemming_word(text):
    text = [stem(word) for word in text.split(" ")]
    return ' '.join(text)


def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known



def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown





def clean_contractions(text):
    mapping=contraction_mapping
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


def clean_stopwords(text):
    # if bad_words:
    #     forbidden_words.update(bad_words)
    pattern = re.compile(r'\b(' + r'|'.join(forbidden_words) + r')\b\s*')
    text = pattern.sub(' ', text)
    return text

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def clean_special_chars(text):
    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])

    for p in punct:
        # text = text.replace(p, f' {p} ')
        text = text.replace(p, f'')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                'है': '', '\\':''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])

    return text


def correct_spelling(x):
    for word in mispell_dict.keys():
        x = x.replace(word, mispell_dict[word])
    return x

def cleaningSpacyObject(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)


# def cleanAnswersForDementiaDataset(dirtyList):
#     bad_words = ['mhm', 'exc', 'uh', 'gram', 'mm', '...', 'uk', "xxx", "laughs", "throat", "um"]
#     i = 0
#     clean_list = []
#     clean_words = []
#     clean_sentences = ""
#     for dirty in dirtyList:
#         sentences = re.split('[.?!]', dirty)
#         for sentence in sentences:
#             sentence = sentence.lower()
#             words = sentence.split()
#             for j in range(len(words)):
#                 if not (words[j] in bad_words):
#                     clean_words.append(words[j])
#             new_sentence = ' '.join(clean_words)
#             clean_words = []
#             new_sentence = new_sentence.replace('(', '')
#             new_sentence = new_sentence.replace(')', ' ')
#             new_sentence = new_sentence.capitalize()
#             if clean_sentences == "":
#                 clean_sentences = new_sentence
#             else:
#                 clean_sentences = clean_sentences + "." + new_sentence
#         clean_list.append(clean_sentences)
#
#         clean_sentences = ""
#         i = i + 1
#     return clean_list