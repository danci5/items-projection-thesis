# edit distance
# word2vec is in prepare_word2vec


def minimum_edit_distance(s1, s2):
    """
    https://en.wikipedia.org/wiki/Levenshtein_distance
    http://rosettacode.org/wiki/Levenshtein_distance#Python

    The Levenshtein distance between two words is the minimum number of single-character edits
    (insertions, deletions or substitutions) required to change one word into the other."""

    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1],
                                             distances[index1 + 1],
                                              new_distances[-1])))
        distances = new_distances

    return distances[-1]


def distance_ratio(s1, s2):
    return minimum_edit_distance(s1, s2) / max(len(s1), len(s2))


def levenshtein_similarity(s1, s2):
    """Basic levenshtein similarity.

    normalization: ed(s1,s2) / max(len(s1),len(s2))
    """
    return 1 - distance_ratio(s1, s2)


def levenshtein_similarity2(s1, s2):
    """Basic levenshtein similarity with another way for normalization.

    normalization: (max(len(s1),len(s2)) - ed(s1,s2)) / max(len(s1),len(s2)
    """
    return (max(len(s1), len(s2)) - minimum_edit_distance(s1, s2)) / (max(len(s1), len(s2)))


if __name__ == '__main__':
    # print(minimum_edit_distance("byl", "nebyl"))
    # print(minimum_edit_distance("ab", "ac"))
    print(levenshtein_similarity("Jones", "Johnson"))
    print(levenshtein_similarity("Paul", "Paul"))

