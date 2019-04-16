import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import scipy


MODULE_NNLM = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"
MODULE_USE = "https://tfhub.dev/google/universal-sentence-encoder/1"

embed = hub.Module(MODULE_NNLM)
sentences = ["i wanna choose multiple rows from 20th to 30th",
             "select cell",
             "select row",
             "delete row",
             "delete cell",
             "format cell",
             "format row",
             "grab item"]
embeddings = embed(sentences)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    embeddings = sess.run(embeddings)

    euclidian = []
    cosine = []
    for i in range(1, len(embeddings)):
        euclidian.append({'sentence': sentences[i], 'idx': i, 'distance': np.linalg.norm(embeddings[0] - embeddings[i])})
        cosine.append({'sentence': sentences[i], 'idx': i, 'distance': scipy.spatial.distance.cosine(embeddings[0], embeddings[i])})

    euclidian = sorted(euclidian, key=lambda x: x['distance'])
    cosine = sorted(cosine, key=lambda x: x['distance'])

    print(sentences[0])
    print('Euclidian Distances: \n')
    for item in euclidian:
        print(item['sentence'], item['distance'])

    print('\n===============\nCosine Distances\n')
    for item in cosine:
        print(item['sentence'], item['distance'])
