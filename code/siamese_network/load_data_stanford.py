import sys
import numpy as np
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path",help="Path where image dataset resides")
parser.add_argument("--save", help = "Path to pickle data to.", default=os.getcwd())
args = parser.parse_args()
data_path = os.path.join(args.path, "python")

train_folder = os.path.join(data_path,'train')
valpath = os.path.join(data_path,'validation')

save_path = args.save

lang_dict = {}

def loadimgs(path,n=0):

    X=[]
    y = []
    category_dict = {}
    curr_y = n
    
    #we load every category seperately so we can isolate them later
    for category in os.listdir(path):
        counter = 0
        if category != '.DS_Store': #Remove .DS_Store file in macOS Directories.
            print("loading category: " + category)
            category_dict[category] = [curr_y,None]
            category_path = os.path.join(path,category)
            category_images=[]
            
            #every image has it's own column in the array, so  load seperately
            for filename in os.listdir(category_path):
                if counter < 20:
                    image_path = os.path.join(category_path, filename)
                    image = Image.open(image_path)
                    a = np.asarray(image)
                    image = Image.fromarray(a)
                    category_images.append(image)
                    y.append(curr_y)

                    curr_y += 1
                    category_dict[category][1] = curr_y - 1

                    counter += 1
                
            try:
                X.append(np.stack(category_images))


            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
        
    y = np.vstack(y)
    X = np.stack(X)
  
    return X,y,category_dict

#Training data

X,y,c=loadimgs(train_folder)
with open(os.path.join(save_path,"train.pickle"), "wb") as f:
	pickle.dump((X,c),f)


#Testing / validation data
X,y,c=loadimgs(valpath)
with open(os.path.join(save_path,"val.pickle"), "wb") as f:
	pickle.dump((X,c),f)
