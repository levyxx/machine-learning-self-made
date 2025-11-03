import sys, os
sys.path.append('..')
sys.path.append(os.pardir)
from dataset import ptb
import numpy as np
from common.util import most_similar, ppmi, create_co_matrix

if __name__ == '__main__':
    window_size = 2
    wordvec_size = 100

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)
    print('counting co-occurence ...')
    C = create_co_matrix(corpus, vocab_size, window_size)
    print('calculating PPMI ...')
    W = ppmi(C, verbose=True)

    print('calculating SVD ...')
    try:
        from sklearn.utils.extmath import randomized_svd
        U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
    except ImportError:
        U, S, V = np.linalg.svd(W)

    word_vecs = U[:, :wordvec_size]

    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)