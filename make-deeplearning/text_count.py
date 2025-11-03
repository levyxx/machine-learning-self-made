import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size, window_size=1)
    W = ppmi(C)

    U, S, V = np.linalg.svd(W)

    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

    plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
    # Headless environments (CI, WSL, remote server) may not support interactive
    # backends, so save the figure to a file instead of showing it.
    out_path = os.path.join(os.getcwd(), 'text_count_plot.png')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.close()
