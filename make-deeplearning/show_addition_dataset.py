import sys, os
sys.path.append(os.pardir)
sys.path.append('..')
from dataset import sequence

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt', seed=1984)
    char_to_id, id_to_char = sequence.get_vocab()

    print(x_train.shape, t_train.shape)
    print(x_test.shape, t_test.shape)

    print(x_train[0])
    print(t_train[0])

    print(''.join([id_to_char[i] for i in x_train[0]]))
    print(''.join([id_to_char[i] for i in t_train[0]]))