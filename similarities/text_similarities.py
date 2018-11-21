from nltk.metrics import edit_distance


def levenshtein_similarity(s1, s2):
    """Computes Levenshtein similarity between two words."""
    return 1 - (edit_distance(s1, s2) / max(len(s1), len(s2)))


def levenshtein_similarity_with_threshold(s1, s2, M=5):
    """Computes Levenshtein similarity between two words with a threshold."""
    return 1 - (min(edit_distance(s1, s2), M) / M)


if __name__ == '__main__':
    print(edit_distance("bylinkář", "bylina"))
    print(edit_distance("křivý", "hřejivý"))
    print(levenshtein_similarity("bylinkář", "bylina"))
    print(levenshtein_similarity("křivý", "hřejivý"))
