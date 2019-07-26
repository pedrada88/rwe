# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import sys
import torch
import random
from preprocessing import load_word_vocab_from_relation_vectors, load_embeddings_filtered_byvocab, load_training_data, split_training_data

#RWE model
class RWE_Model(torch.nn.Module):
    def __init__(self, embedding_size_input, embedding_size_output, embedding_weights,hidden_size,dropout):
        super(RWE_Model, self).__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(embedding_weights).float()
        self.embeddings.weight.requires_grad = True
        self.linear1 = torch.nn.Linear(embedding_size_input*2, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(hidden_size, embedding_size_output)
    def forward(self,input1,input2):
        embed1 = self.embeddings(input1)
        embed2 = self.embeddings(input2)
        out = self.linear1(torch.cat(((embed1*embed2), (embed1+embed2)/2), 2)).squeeze()
        out = self.relu(out)
        out = self.dropout(out)
        out= self.linear2(out)
        return out


#Initialize RWE model
def getRWEModel(embedding_size_input, embedding_size_output, embedding_weights,hidden_size,dropout):
    vocab_size=(len(embedding_weights))
    model=RWE_Model(embedding_size_input, embedding_size_output, embedding_weights,hidden_size,dropout)
    criterion = torch.nn.MSELoss()
    return model.cuda(), criterion

#Train epochs
def trainEpochs(model, optimizer, criterion, trainBatches, validBatches, epochs=10, interval=100, lr=0.1):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2, threshold = 1e-7, factor = 0.9)
    min_error=-1.0
    for epoch in range(1, epochs+1):
        print("\n     ----------    \n")
        print ("EPOCH "+str(epoch))
        print ("Starting training epoch "+str(epoch))
        trainIntervals(model, optimizer, criterion, trainBatches, interval, lr)
        validErr = validate(model, validBatches, criterion)
        scheduler.step(validErr)
        print("Validation error : " + str(validErr))
        if validErr<min_error or min_error==-1.0:
            new_model=model
            min_error=validErr
            print ("[Model at epoch "+str(epoch)+" obtained the lowest development error rate so far.]")
        #if epoch%5==0 or epoch==1: torch.save(model, "./"+outputModelFile + "-epoch" + str(epoch) + ".model")
        print("Epoch " + str(epoch) + " done")
    return new_model

#Train intervals
def trainIntervals(model, optimizer, criterion, batches, interval=100, lr=0.1):
    i = 0
    n = 0
    trainErr = 0
    for x1, x2, y in zip(*batches):
        model.train(); optimizer.zero_grad()
        trainErr += gradUpdate(model, x1, x2, y, criterion, optimizer, lr)
        i += 1
        if i == interval:
            n += 1;
            prev_train_err = trainErr
            trainErr = 0
            i = 0
    if i > 0 and prev_train_err != 0:
        print("Training error: "+ str(prev_train_err / float(i)))

#Validation phase
def validate(model, batches, criterion):
    evalErr = 0
    n = 0
    model.eval()
    for x1, x2, y in zip(*batches):
        y = torch.autograd.Variable(y, requires_grad=False)
        x1 = torch.autograd.Variable(x1, requires_grad=False)
        x2 = torch.autograd.Variable(x2, requires_grad=False)
        output = model(x1, x2)
        error = criterion(output, y)
        evalErr += error.item()
        n += 1
    return evalErr / n

#Update gradient
def gradUpdate(model, x1, x2, y, criterion, optimizer, lr):
    output = model(x1,x2)
    error = criterion(output,y)
    error.backward()
    optimizer.step()
    return error.item()

#Get batches from training set
def getBatches(data, batchSize):
    embsize = int(data.size(-1))
    return data.view(-1, batchSize, embsize) 


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-word_embeddings', '--input_word_embeddings', help='Input word embeddings path', required=True)
    parser.add_argument('-rel_embeddings', '--input_relation_embeddings', help='Input relation embeddings path', required=True)
    parser.add_argument('-output', '--output_path', help='Output path to store the output relational word embeddings', required=True)
    parser.add_argument('-hidden', '--hidden_size', help='Size of the hidden layer (default=2xdimensions-wordembeddings)', required=False, default=0)
    parser.add_argument('-dropout', '--drop_rate', help='Dropout rate', required=False, default=0.5)
    parser.add_argument('-epochs', '--epochs_num', help='Number of epochs', required=False, default=5)
    parser.add_argument('-interval', '--interval', help='Size of intervals during training', required=False, default=100)
    parser.add_argument('-batchsize', '--batchsize', help='Batch size', default=10)
    parser.add_argument('-devsize', '--devsize', help='Size of development data (proportion with respect to the full training set, from 0 to 1)', required=False, default=0.015)
    parser.add_argument("-lr", '--learning_rate', help='Learning rate for training', required=False, default=0.01)

    args = vars(parser.parse_args())
    word_embeddings_path=args['input_word_embeddings']
    rel_embeddings_path=args['input_relation_embeddings']
    output_path=args['output_path']
    hidden_size=int(args['hidden_size'])
    dropout=float(args['drop_rate'])
    epochs=int(args['epochs_num'])
    interval=int(args['interval'])
    batchsize=int(args['batchsize'])
    devsize=float(args['devsize'])
    lr=float(args['learning_rate'])
    if devsize>=1 or devsize<0: sys.exit("Development data should be between 0% (0.0) and 100% (1.0) of the training data")

    print ("Loading word vocabulary...")
    pre_word_vocab=load_word_vocab_from_relation_vectors(rel_embeddings_path)
    print ("Word vocabulary loaded succesfully ("+str(len(pre_word_vocab))+" words). Now loading word embeddings...")
    matrix_word_embeddings,word2index,index2word,dims_word=load_embeddings_filtered_byvocab(word_embeddings_path,pre_word_vocab)
    pre_word_vocab.clear()
    print ("Word embeddings loaded succesfully ("+str(dims_word)+" dimensions). Now loading relation vectors...")
    matrix_input,matrix_output,dims_rels=load_training_data(rel_embeddings_path,matrix_word_embeddings,word2index)
    print ("Relation vectors loaded ("+str(dims_rels)+" dimensions), now spliting training and dev...")
    random.seed(21)
    s1 = random.getstate()
    random.shuffle(matrix_input)
    random.setstate(s1)
    random.shuffle(matrix_output)
    matrix_input_train,matrix_output_train,matrix_input_dev,matrix_output_dev=split_training_data(matrix_input,matrix_output,devsize,batchsize)
    matrix_input.clear()
    matrix_output.clear()
    print ("Done preprocessing all the data, now loading and training the model...\n")
    
    if hidden_size==0: hidden_size=dims_word*2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ("Device used: "+str(device))
    embedding_weights=torch.tensor(matrix_word_embeddings)
    matrix_word_embeddings.clear()
    tensor_input_train_1=torch.LongTensor([[x[0]] for x in matrix_input_train])
    tensor_input_train_2=torch.LongTensor([[x[1]] for x in matrix_input_train])
    matrix_input_train.clear()
    tensor_input_dev_1=torch.LongTensor([[x[0]] for x in matrix_input_dev])
    tensor_input_dev_2=torch.LongTensor([[x[1]] for x in matrix_input_dev])
    matrix_input_dev.clear()
    tensor_output_train=torch.FloatTensor(matrix_output_train)
    matrix_output_train.clear()
    tensor_output_dev=torch.FloatTensor(matrix_output_dev)
    matrix_output_dev.clear()
    model, criterion = getRWEModel(dims_word,dims_rels,embedding_weights,hidden_size,dropout)
    print ("RWE model loaded.")
    optimizer = torch.optim.Adam(model.parameters(), lr)
    trainX1batches = getBatches(tensor_input_train_1.cuda(), batchsize)
    trainX2batches = getBatches(tensor_input_train_2.cuda(), batchsize)
    validX1Batches = getBatches(tensor_input_dev_1.cuda(), batchsize)
    validX2Batches = getBatches(tensor_input_dev_2.cuda(), batchsize)
    trainYBatches = getBatches(tensor_output_train.cuda(), batchsize)
    validYBatches = getBatches(tensor_output_dev.cuda(), batchsize)
    print ("Now starting training...\n")
    output_model=trainEpochs(model, optimizer, criterion, (trainX1batches, trainX2batches, trainYBatches), (validX1Batches, validX2Batches, validYBatches), epochs, interval, lr)
    print ("\nTraining finished. Now loading relational word embeddings from trained model...")
    
    parameters=list(output_model.parameters())
    num_vectors=len(parameters[0])
    print ("Number of vectors: "+str(num_vectors))
    num_dimensions=len(parameters[0][0])
    print ("Number of dimensions output embeddings: "+str(num_dimensions))
    txtfile=open(output_path,'w',encoding='utf8')
    txtfile.write(str(num_vectors)+" "+str(num_dimensions)+"\n")
    if num_vectors!=len(matrix_word_embeddings): print ("Something is wrong in the input vectors: "+str(len_vectors)+" != "+str(num_vectors))
    for i in range(num_vectors):
        word=index2word[i]
        txtfile.write(word)
        vector=parameters[0][i].cpu().detach().numpy()
        for dimension in vector:
            txtfile.write(" "+str(dimension))
        txtfile.write("\n")
    txtfile.close()
    print ("\nFINISHED. Word embeddings stored at "+output_path)


            
        

    
    
    
    
    
