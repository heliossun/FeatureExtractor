# Valentin Mace
# valentin.mace@kedgebs.com
# Developed at Qwant Research

"""Functions adding noise to text"""

import random
from nltk.tokenize import word_tokenize

def random_bool(probability=0):
    """Returns True with given probability

    Args:
        probability: probability to return True

    """
    assert (0 <= probability <= 1), "probability needs to be >= 0 and <= 1"
    return random.random() < probability

def delete_random_token(line, probability=0.2):
    """Delete random tokens in a given String with given probability

    Args:
        line: a String
        probability: probability to delete each token

    """
    line_split = line.split()
    ret = [token for token in line_split if not random_bool(probability)]

    return " ".join(ret)


def replace_random_token(line, probability=0.2, filler_token="BLANK"):
    """Replace random tokens in a String by a filler token with given probability

    Args:
        line: a String
        probability: probability to replace each token
        filler_token: token replacing chosen tokens

    """
    line_split = line.split()
    for i in range(len(line_split)):
        if random_bool(probability):
            line_split[i] = filler_token

    return " ".join(line_split)


def random_token_permutation(line, _range=2):
    """Random permutation over the tokens of a String, restricted to a range, drawn from the uniform distribution

    Args:
        line: a String
        _range: Max range for token permutation

    """
    line_split = line.split()
    new_indices = [i+random.uniform(0, _range+1) for i in range(len(line_split))]
    res = [x for _, x in sorted(zip(new_indices, line_split), key=lambda pair: pair[0])]
    return " ".join(res)
