import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import jieba

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import networkx as nx

def get_connect_graph_by_text_rank(tokenized_text, window=3):
    keywords_graph = nx.Graph()
    tokeners = tokenized_text.split()
    for ii, t in enumerate(tokeners):
        word_tuples = [(tokeners[connect], t) 
                       for connect in range(ii-window, ii+window+1) 
                       if connect >= 0 and connect < len(tokeners)]
        keywords_graph.add_edges_from(word_tuples)

    return keywords_graph

def TextRank(cut_sents):
    #connect graph
    words_graph = get_connect_graph_by_text_rank(" ".join(cut_sents[0]))

    #draw networkx
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(3, figsize=(12, 12))
    nx.draw_networkx(words_graph, font_size=12)

    #get dictory
    g_dic = {}
    node_set = set()
    for i, j in words_graph.edges():
        node_set.add(i)
        node_set.add(j)
        if i == j: continue
        if i not in g_dic : g_dic[i] = set()
        g_dic[i].add(j)

    #matrics
    node_mat = np.zeros((len(node_set), len(node_set)))
    for i, w1 in enumerate(g_dic):
        p = 1/len(g_dic[w1])
        for j, w2 in enumerate(g_dic[w1]):
            node_mat[j][i] = p
    node_mat

    #text rank
    d = 0.85
    pr = np.ones((len(node_set), 1)) 
    for i in range(10):
        print(pr.T)
        pr = 1/len(node_set) * (1-d) + d * np.dot(node_mat,pr)
    print('\n',pr.T)