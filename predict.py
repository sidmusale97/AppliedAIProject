import argparse
import tensorflow as tf
import ctc_utils
from cv2 import cv2
import numpy as np

notes_dict = {'L-1': 57 , 'L0': 60, 'S0': 62, 'L1': 64, 'S1': 65, 'L2': 67, 'S2': 69, 'L3': 71, 'S3': 72, 'L4': 75, 'S4': 76, 'L5': 77}

def predict(image):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    voc_file = 'vocabulary_agnostic.txt'
    model = './Models/model.hdf5-69000.meta'

    # Read the dictionary
    dict_file = open(voc_file, "r")
    dict_list = dict_file.read().splitlines()
    int2word = dict()
    for word in dict_list:
        word_idx = len(int2word)
        int2word[word_idx] = word
    dict_file.close()

    # Restore weights
    saver = tf.train.import_meta_graph(model)
    saver.restore(sess, model[:-5])

    graph = tf.get_default_graph()

    input = graph.get_tensor_by_name("model_input:0")
    seq_len = graph.get_tensor_by_name("seq_lengths:0")
    rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
    height_tensor = graph.get_tensor_by_name("input_height:0")
    width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
    logits = tf.get_collection("logits")[0]

    # Constants that are saved inside the model itself
    WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

    decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

    image = cv2.imread(image, False)
    image = ctc_utils.resize(image, HEIGHT)
    image = ctc_utils.normalize(image)
    image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)

    seq_lengths = [image.shape[2] / WIDTH_REDUCTION]

    prediction = sess.run(
        decoded, feed_dict={input: image, seq_len: seq_lengths, rnn_keep_prob: 1.0,}
    )

    str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
    notes = []
    for w in str_predictions[0]:
        temp = int2word[w].split('.')
        print(temp)
        if (len(temp) != 2):
            continue
        else:
            symbol, des = temp       
            if (symbol == 'note'):
                length, note = des.split('-', 1)
                if ('beamed' in length):
                    length = 'eigth'
                notes.append((length, notes_dict[note]))
            elif (symbol == 'rest'):
                length, _ = des.split('-', 1)
                notes.append((length, 'rest'))

    return notes

print(predict('IMG_1954.png'))