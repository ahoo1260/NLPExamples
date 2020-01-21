
from nltk import pos_tag
from nltk.corpus import wordnet as wn

ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'


#data is a list of list. each list is a sentence
def get_pos_tags(data):
    return [pos_tag(sent) for sent in data]


def if_pos_is_frequent(desired_pos, all_pos):
    frequent=0
    for pos in all_pos:
        if desired_pos==pos:
            frequent=frequent+1

    if frequent>len(all_pos)/2:
        return True
    else:
        return False


def is_noun(word):
    if wn.synsets(word):
        all_pos=[o.pos() for o in wn.synsets(word)]
        if if_pos_is_frequent(NOUN, all_pos):
            return True
        else:
            return False


def is_verb(word):
    if wn.synsets(word):
        all_pos = [o.pos() for o in wn.synsets(word)]
        if if_pos_is_frequent(VERB, all_pos):
            return True
        else:
            return False

def is_adjective(word):
    if wn.synsets(word):
        all_pos = [o.pos() for o in wn.synsets(word)]
        if if_pos_is_frequent(ADJ, all_pos):
            return True
        else:
            return False


def is_adverb(word):
    if wn.synsets(word):
        tmp = wn.synsets(word)[0].pos()
        if tmp==ADV:
            return True
        else:
            return False