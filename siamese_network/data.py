import os
import glob
import numpy as np
import random
from imageutils import load_image
import tensorflow as tf



class Data:
    def __init__(self):
        self.name = "siamese_datagen"

    def create_positive_pairs(self,x,class_indices,number_pairs):
        
        positive_pairs = []
        positive_labels = []
        _classes = list(class_indices.keys())
        
        for _ in range(int(number_pairs)):
            cls_1 = random.choice(_classes)
            cls1_elements = list(class_indices[cls_1][0])
            element_index_1, element_index_2 = self.get_random_unequal_index(cls1_elements)

            positive_pairs.append([load_image(x[element_index_1]),load_image(x[element_index_2])])
            
            """
            below line is written to view whether the pairs are chosen correctly comment 
            the above line and uncomment the below to check
            """
            # positive_pairs.append([x[element_index_1],x[element_index_2]])
            positive_labels.append([1.0])
        
        return positive_pairs,positive_labels

    def create_negative_pairs(self,x,class_indices,number_pairs):

        negative_pairs = []
        negative_labels = []
        _classes = list(class_indices.keys())

        for _ in range(int(number_pairs)):
            cls_1, cls_2 = self.get_random_unequal_index(_classes)
            cls1_elements = list(class_indices[cls_1][0])
            cls2_elements = list(class_indices[cls_2][0])

            element_index_1 = random.choice(cls1_elements)
            element_index_2 = random.choice(cls2_elements)

            negative_pairs.append([load_image(x[element_index_1]),load_image(x[element_index_2])])
            
            """
            below line is written to view whether the pairs are chosen correctly comment 
            the above line and uncomment the below to check
            """
            # negative_pairs.append([x[element_index_1],x[element_index_2]])
            negative_labels.append([0.0])
        
        return negative_pairs,negative_labels


    def create_pairs(self,x,class_indices,batch_size):
        
        num_pairs = batch_size / 2
        positive_pairs, positive_labels = self.create_positive_pairs(x, class_indices, num_pairs)
        negative_pairs, negative_labels = self.create_negative_pairs(x, class_indices, num_pairs)
        return np.array(positive_pairs + negative_pairs),np.array(positive_labels + negative_labels)

    def get_class_indices(self,y):
        
        _classes = np.unique(y)
        _num_classes = len(_classes)
        class_indices = {}
        for _each_class in _classes:
            indices = np.where(y == _each_class)
            class_indices[_each_class]= indices
        
        return class_indices, _num_classes


    def getDataSet(self,data_dir):
        class_labels = []
        images_list = glob.glob(os.path.join(data_dir,"*.jpg"))
        images_list.extend(glob.glob(os.path.join(data_dir,"*.png")))
        for _image_path in images_list:
            class_label = os.path.basename(_image_path)[:4]
            class_labels.append(class_label)

        return np.asarray(images_list),np.asarray(class_labels)

    
    def pairs_data_generator(self,x,y,batch_size):
        class_indices, num_classes =  self.get_class_indices(y)
        while True:
            pairs,labels = self.create_pairs(x,class_indices,batch_size)

            # sending one to input_1 and other img in pair to input 2 and labels to label
            yield [pairs[:, 0].squeeze(), pairs[:, 1].squeeze()], labels
     
        
    def get_random_unequal_index(self,_list):
        index1,index2 = random.choices(_list,k=2)
        while index1 == index2:
            index1,index2 = random.choices(_list,k=2)

        return index1,index2


if __name__=="__main__":
    # data = Data()
    # x,y = data.getDataSet("~/Documents/Project_outsource/archive/Market-1501-v15.09.15/query")
    
    # ind,num = data.get_class_indices(y)

    # pairs, labels = data.create_pairs(x,ind,4)

    # print([pairs[:, 0].squeeze().shape, pairs[:, 1].squeeze().shape])
    pass
