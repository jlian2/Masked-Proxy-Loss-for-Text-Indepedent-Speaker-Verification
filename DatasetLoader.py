#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import random
import pdb
import os
import threading
import time
import math
from scipy.io import wavfile
from queue import Queue

def round_down(num, divisor):
    return num - (num%divisor)


def loadWAV(filename, max_frames, evalmode=True, num_eval=20):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio  = wavfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
        audio       = numpy.pad(audio, (shortage, shortage), 'constant', constant_values=0)
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0)

    feat = torch.FloatTensor(feat)

    return feat;

class DatasetLoader(object):
    def __init__(self, dataset_file_name, batch_size, max_frames, max_seg_per_spk, nDataLoaderThread, gSize, train_path, maxQueueSize = 10, **kwargs):
        self.dataset_file_name = dataset_file_name;
        self.nWorkers = nDataLoaderThread;
        self.max_frames = max_frames;
        self.max_seg_per_spk = max_seg_per_spk;
        self.batch_size = batch_size;
        self.maxQueueSize = maxQueueSize;

        self.data_dict = {};
        self.data_list = [];
        self.nFiles = 0;
        self.gSize  = gSize; ## number of clips per sample (e.g. 1 for softmax, 2 for triplet or pm)

        self.dataLoaders = [];
        
        ### Read Training Files...
        with open(dataset_file_name) as dataset_file:
            while True:
                line = dataset_file.readline();
                if not line:
                    break;
                
                data = line.split();
                speaker_name = data[0];
                filename = os.path.join(train_path,data[1]);

                if not (speaker_name in self.data_dict):
                    self.data_dict[speaker_name] = [];

                self.data_dict[speaker_name].append(filename);

        ### Initialize Workers...
        self.datasetQueue = Queue(self.maxQueueSize);
    

    def dataLoaderThread(self, nThreadIndex):
        
        index = nThreadIndex*self.batch_size;

        if(index >= self.nFiles):
            return;

        while(True):
            if(self.datasetQueue.full() == True):
                time.sleep(1.0);
                continue;

            in_data = [];
            for ii in range(0,self.gSize):
                feat = []
                for ij in range(index,index+self.batch_size):
                    feat.append(loadWAV(self.data_list[ij][ii], self.max_frames, evalmode=False));
                
                #print(feat[0].shape)  #(1,32240)
                in_data.append(torch.cat(feat, dim=0));
                
                #print(in_data[0].shape) #(batch, 32240)

            in_label = numpy.asarray(self.data_label[index:index+self.batch_size]);
            
            self.datasetQueue.put([in_data, in_label]);
            
            #print(len(in_data))    2 for gSize=2

            index += self.batch_size*self.nWorkers;  #here is how thread works

            if(index+self.batch_size > self.nFiles):
                break;



    def __iter__(self): #first: update dataloader(give whole dataset at one time)  (iterator typically returns itself)

        dictkeys = list(self.data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        
        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data    = self.data_dict[key] #e.g. len(data) = 103
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.gSize) #numSeg = 102
            rp      = lol(numpy.random.permutation(len(data))[:numSeg],self.gSize) #(51, 2)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])
        
        #flattened_list: [(51,2),(23,2),(40,2),...,(19,2),(13,2)]  len(flattened_list) = 900000
        #flattened_label: [(0,0,0,0,...),(1,1,1,1,...),(2,2,2,2,2....),......,(5992,5992,5992,5992,....,5992),(5993,5993,5993,...,5993)]
        
        #another manner:
        #flattened_list: [(34,3),(23,2),(27,3),...,(13,3),(13,2)]  len(flattened_list) = 900000
        #flattened_label: [(0,0,0,0,...),(1,1,1,1,...),(2,2,2,2,2....),......,(5992,5992,5992,5992,....,5992),(5993,5993,5993,...,5993)]
        
        ## Data in random order
        mixid           = numpy.random.permutation(len(flattened_label))
        
        
        mixlabel        = []
        mixmap          = []

        
        #mixid = [55,107,200,500,2,10,17777,29999,1232,8,.......,....] (len(mixid) = 900000)
        #mixlabel = [1,2,5,10,0,...]
        #mixmap = [55,107,200,500,...]
        
        
        ## Prevent two pairs of the same speaker in the same batch
        
        #e.g. batch_size = 128
        #len(mixlabel) <= 128 
        #len(mixlabel) = 133: startbatch = (133-5=128)
        
        #only focus on the current batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)
                
        
                
        ## This is for MP and MMP
        #for ii in mixid:
        #    startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
        #    mixlabel.append(flattened_label[ii])
        #    mixmap.append(ii)
                

        self.data_list  = [flattened_list[i] for i in mixmap] #this is final data that does not have same labels in the same batch
        self.data_label = [flattened_label[i] for i in mixmap]
        
        
        
        ## Iteration size
        self.nFiles = len(self.data_label); #900000

        ### Make and Execute Threads...
        for index in range(0, self.nWorkers):  #so that num_workers * batch?
            self.dataLoaders.append(threading.Thread(target = self.dataLoaderThread, args = [index]));
            self.dataLoaders[-1].start(); #put batch data and label into queue

        return self;


    def __next__(self):

        while(True):
            isFinished = True;
            
            if(self.datasetQueue.empty() == False):
                return self.datasetQueue.get(); #one batch
            for index in range(0, self.nWorkers):
                if(self.dataLoaders[index].is_alive() == True):
                    isFinished = False;
                    break;

            if(isFinished == False):
                time.sleep(1.0);
                continue;


            for index in range(0, self.nWorkers):
                self.dataLoaders[index].join();

            self.dataLoaders = [];
            raise StopIteration;


    def __call__(self):
        pass;

    def getDatasetName(self):
        return self.dataset_file_name;

    def qsize(self):
        return self.datasetQueue.qsize();