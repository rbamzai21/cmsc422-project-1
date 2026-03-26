import numpy as np
from collections import Counter


class TennisData:
    #                 Outlook      Temperature   Hum   Wind
    #                S?  O?  R?     H?  M?  C?    H?    S?
    X = np.array([[  1,  0,  0,     1,  0,  0,    1,    0   ],
                  [  1,  0,  0,     1,  0,  0,    1,    1   ],
                  [  0,  1,  0,     1,  0,  0,    1,    0   ],
                  [  0,  0,  1,     0,  1,  0,    1,    0   ],
                  [  0,  0,  1,     0,  0,  1,    0,    0   ],
                  [  0,  0,  1,     0,  0,  1,    0,    1   ],
                  [  0,  1,  0,     0,  0,  1,    0,    1   ],
                  [  1,  0,  0,     0,  1,  0,    1,    0   ],
                  [  1,  0,  0,     0,  0,  1,    0,    0   ],
                  [  0,  0,  1,     0,  1,  0,    0,    0   ],
                  [  1,  0,  0,     0,  1,  0,    0,    1   ],
                  [  0,  1,  0,     0,  1,  0,    1,    1   ],
                  [  0,  1,  0,     1,  0,  0,    0,    0   ],
                  [  0,  0,  1,     0,  1,  0,    1,    1   ]
                  ], dtype=float)

    Y = np.array([ -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1 ], dtype=float)

    #                   Outlook      Temperature   Hum   Wind
    #                  S?  O?  R?     H?  M?  C?    H?    S?
    Xte = np.array([[  1,  0,  0,     1,  0,  0,    1,    0   ],
                    [  1,  0,  0,     1,  0,  0,    1,    1   ],
                    [  0,  0,  1,     0,  0,  1,    0,    0   ],
                    [  0,  1,  0,     0,  0,  1,    0,    1   ],
                    [  1,  0,  0,     0,  0,  1,    0,    0   ],
                    [  0,  0,  1,     0,  1,  0,    1,    1   ]
                    ], dtype=float)

    Yte = np.array([ -1, -1, 1, 1, 1, -1 ], dtype=float)


def load_text_data(filename):
    wfreq = Counter()
    h = open(filename, "r")
    d = []
    meta = []
    for l in h.readlines():
        meta_split = l.strip().split("\t")
        a = meta_split[0].split()
        if len(meta_split) > 1:
            meta.append(meta_split[1])
        else:
            meta.append("")
        if len(a) > 1:
            y = float(a[0])
            if y > 0.5: y = 1.
            else: y = -1.
            x = {}
            for w in a[1:]:
                x[w] = x.get(w,0) + 1.
            for w in x.keys():
                wfreq[w] += 1
            d.append( (x,y) )
    h.close()

    wid = {}
    widr = []
    max_id = 1
    for w, c in wfreq.items():
        if c >= 10 and c < 0.7*len(d):
            wid[w] = max_id
            widr.append(w)
            max_id += 1

    n = len(d)

    x_all = np.zeros((n, max_id-1), dtype=float)
    y_all = np.zeros((n,), dtype=float)
    for i in range(len(d)):
        (x, y) = d[i]
        y_all[i] = y
        for w in x.keys():
            if w in wid:
                x_all[i, wid[w]-1] = 1.

    return x_all, y_all, widr, np.array(meta)

        
class SentimentData:
    Xall, Yall, words, meta = load_text_data("data/sentiment.all")
    N, D = Xall.shape
    N0 = int(float(N) * 0.6)
    N1 = int(float(N) * 0.8)
    X = Xall[0:N0,:]
    Y = Yall[0:N0]
    Xde = Xall[N0:N1,:]
    Yde = Yall[N0:N1]
    Xte = Xall[N1:,:]
    Yte = Yall[N1:]
