import collections
import pickle
import time
import math.log


# def tfidf_vectorizer(vector, maxtf = True):
#     """Transforms a raw count vector into a tfidf vector using maxtf by default, original tfcounts otherwise"""
#     print('recomputing tfidf values for industries...')
#     start_time = time.clock()
#     tfidf_values = collections.defaultdict(dict)
#     alldocs = len(vector)
#     for key in vector:
#         for gram in vector[key]:
#             raw_tf = vector[key][gram]
#             df = len([vector[sector][gram] for sector in vector if gram in vector[sector]]) + 0.001
#             if maxtf:
#                 maxtf = max([vector[key][gram] for gram in vector[key]])
#                 maxtf_norm = 0.4 + ((1 - 0.4) * (raw_tf / maxtf))
#                 max_tfidf = maxtf_norm * math.log(alldocs / df)
#                 tfidf_values[key][gram] = max_tfidf
#             else:
#                 tfidf = raw_tf * math.log(alldocs / df)
#                 tfidf_values[key][gram] = tfidf
#     print('done.')
#     fnames = {1:'tfidf_ivector.data', 2: 'tfidf_cvector.data'}
#     filename = fnames[i]
#     pickle.dump(tfidf_values, open(filename, "wb" ))
#     print( "%.2f" % (time.clock() - start_time), "seconds")
#     return tfidf_values
