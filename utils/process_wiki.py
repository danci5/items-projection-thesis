"""
Usage: python process_wiki.py cswiki-latest-pages-articles.xml.bz2 wiki.cs.text

Script for converting articles from a Wikipedia dump to a file. A line in that file is a text of an article.

Download latest CS Wikipedia dump from:
- https://dumps.wikimedia.org/cswiki/latest/cswiki-latest-pages-articles.xml.bz2
Adapted from:
- http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
"""

import logging
import os.path
import sys
from gensim.corpora import WikiCorpus


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

    i = 0
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={}, lower=False)
    with open(outp, 'w') as out:
        for text in wiki.get_texts():
            out.write(" ".join(text) + "\n")
            i += 1
            if i % 10000 == 0:
                logger.info("Saved %s articles", i)

    logger.info("Finished saving %s articles", i)
