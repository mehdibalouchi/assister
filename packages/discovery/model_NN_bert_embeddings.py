# from bert_embedding import BertEmbedding
from bert_serving.client import BertClient
import numpy as np
import scipy.spatial

def normalized_correlation_coefficient(sequence, template):
    pad_size_0 = int(template.shape[0] / 2)

    padded_sequence = np.pad(sequence, (pad_size_0, pad_size_0), mode='constant')
    correlated_sequence = np.zeros_like(sequence).astype(float)
    for r in range(correlated_sequence.shape[0]):

        overlapping_padded_seq = padded_sequence[r:r + template.shape[0]]
        correlated_sequence[r] = np.corrcoef(overlapping_padded_seq, template)[0, 1]
    return correlated_sequence

sentences = ["i wanna choose multiple rows from 20th to 30th",
             "select cell",
             "select row",
             "delete row",
             "delete cell",
             "format cell",
             "format row",
             "grab item"]

bert_client = BertClient()
embeddings = bert_client.encode(sentences)

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