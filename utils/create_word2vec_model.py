"""
Usage: python create_word2vec_model.py wiki.cs.text word2vec.model

Script for creating/training a word2vec model.
"""

import logging
import multiprocessing
import os
import sys
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("Running %s", ' '.join(sys.argv))

    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    inp, outp = sys.argv[1:3]
    params = {
        'size': 200,
        'window': 5,
        'min_count': 5,
        'workers': multiprocessing.cpu_count() - 1
    }
    model = Word2Vec(LineSentence(inp), **params)
    model.save(outp)
