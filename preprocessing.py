# -*- coding: utf-8 -*-
import numpy as np
import random

#Load embedding vocabulary
def load_vocab_embeddings(input_path):
    first_line=True
    vocab=set()
    input_file_relations=open(input_file_relations, 'r', encoding='utf-8')
    for line in input_file_relations:
        if first_line==True:
            first_line=False
        else:
            vocab.add(line.strip().split(" ")[0])
    return vocab

#Load embedding vocabulary
def load_word_vocab_from_relation_vectors(input_path):
    pre_word_vocab=set()
    first_line=True
    final_word_vocab=set()
    input_file_relations=open(input_path, 'r', encoding='utf-8')
    for line in input_file_relations:
        linesplit=line.strip().split(" ")
        if first_line==True:
            first_line=False
        else:
            relation=linesplit[0]
            if "__" not in relation: sys.exit("ERROR: Pair '"+relation+"' does not contain underscore")
            relation_split=relation.rsplit("__",1)
            word1=relation_split[0]
            word2=relation_split[1]
            pre_word_vocab.add(word1)
            pre_word_vocab.add(word2)
    return pre_word_vocab


#Load embeddings filtered by pre-given vocabulary
def load_embeddings_filtered_byvocab(input_path,vocab):
    word2index={}
    index2word={}
    matrix_word_embeddings=[]
    first_line=True
    input_file_relations=open(input_path, 'r', encoding='utf-8')
    cont=0
    for line in input_file_relations:
        linesplit=line.strip().split(" ")
        if first_line==True:
            dimensions=int(linesplit[1])
            first_line=False
        else:
            word=linesplit[0]
            if word in vocab and word not in word2index:
                word2index[word]=cont
                index2word[cont]=word
                cont+=1
                matrix_word_embeddings.append(np.asarray([float(dim) for dim in linesplit[1:dimensions+1]]))
    return matrix_word_embeddings,word2index,index2word,dimensions

#Load embedding matrices input/output
def load_training_data(input_path,matrix_word_embeddings,word2index):
    matrix_input=[]
    matrix_output=[]
    first_line=True
    input_file_relations=open(input_path, 'r', encoding='utf-8')
    for line in input_file_relations:
        linesplit=line.strip().split(" ")
        if first_line==True:
            dimensions=int(str(line.split(" ")[1]))
            first_line=False
        else:
            relation=linesplit[0]
            if "__" not in relation: sys.exit("ERROR: Pair '"+relation+"' does not contain underscore")
            relation_split=relation.rsplit("__",1)
            word1=relation_split[0]
            word2=relation_split[1]
            if word1 in word2index and word2 in word2index:
                matrix_input.append(np.asarray([word2index[word1],word2index[word2]]))
                matrix_output.append(np.asarray([float(dim) for dim in linesplit[1:dimensions+1]]))
    return matrix_input,matrix_output,dimensions

#Split training and development data
def split_training_data(matrix_input,matrix_output,devsize,batchsize):
    matrix_input_train=[]
    matrix_output_train=[]
    matrix_input_dev=[]
    matrix_output_dev=[]
    num_instances=int((len(matrix_input)//batchsize)*batchsize)
    final_size_dev=int(((num_instances*devsize)//batchsize)*batchsize)
    final_size_train=int(((num_instances-final_size_dev)//batchsize)*batchsize)
    print ("Size train set: "+str(final_size_train))
    print ("Size dev set: "+str(final_size_dev))
    all_instances=range(num_instances)
    list_index_dev=random.sample(all_instances,final_size_dev)
    for i in range(num_instances):
        if i in list_index_dev:
            matrix_input_dev.append(matrix_input[i])
            matrix_output_dev.append(matrix_output[i])
        else:
            matrix_input_train.append(matrix_input[i])
            matrix_output_train.append(matrix_output[i])
    return matrix_input_train,matrix_output_train,matrix_input_dev,matrix_output_dev
