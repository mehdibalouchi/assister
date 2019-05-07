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


sentences = ["i",
             "wanna",
             "choose",
             "multiple",
             "rows",
             "from",
             "20th",
             "to",
             "30th",
             "delete",
             "row",
             "cell",
             "select",
             "format",
             "grab",
             "item"]


bert_client = BertClient()
embeddings = bert_client.encode(sentences)

sentence = np.concatenate(embeddings[0:-7])

templates = []
template_1 = np.concatenate([embeddings[-4], embeddings[-5]])  # select cell

template_2 = np.concatenate([embeddings[-4], embeddings[-6]])  # select row
template_3 = np.concatenate([embeddings[-7], embeddings[-6]])  # delete row
template_4 = np.concatenate([embeddings[-7], embeddings[-5]])  # delete cell
template_5 = np.concatenate([embeddings[-3], embeddings[-5]])  # format cell
template_6 = np.concatenate([embeddings[-3], embeddings[-6]])  # format row
template_7 = np.concatenate([embeddings[-2], embeddings[-1]])  # grab item

corr_seq_1 = normalized_correlation_coefficient(sentence, template_1)
corr_seq_2 = normalized_correlation_coefficient(sentence, template_2)
corr_seq_3 = normalized_correlation_coefficient(sentence, template_3)
corr_seq_4 = normalized_correlation_coefficient(sentence, template_4)
corr_seq_5 = normalized_correlation_coefficient(sentence, template_5)
corr_seq_6 = normalized_correlation_coefficient(sentence, template_6)
corr_seq_7 = normalized_correlation_coefficient(sentence, template_7)

print(' '.join(sentences[:-7]))
print('Maximum normalized correlation coefficient between query and template')
print('select cell', np.max(corr_seq_1))
print('select row', np.max(corr_seq_2))
print('delete row', np.max(corr_seq_3))
print('delete cell', np.max(corr_seq_4))
print('format cell', np.max(corr_seq_5))
print('format row', np.max(corr_seq_6))
print('grab item', np.max(corr_seq_7))