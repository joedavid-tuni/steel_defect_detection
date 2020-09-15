import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from utils import mask2contour
from keras import backend as K
import segmentation_models as sm
from os import listdir
from PIL import Image
import argparse
import sys


def dice_coef(y_true, y_pred, smooth=1):
    """Calculates the dice coefficient

    Args:
        y_true: pixel truth values
        y_pred: pixel predicted values
        smooth: smooth parameter to smooth the loss function

    Returns:
        (float): Returns the dice coefficient
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def get_model():
    """Gets the model used for training the dataset

    Returns:
        model: the model used for training
        preprocess: the preprocessing model
    """

    preprocess = sm.get_preprocessing('resnet34')
    model = sm.Unet('resnet34', input_shape=(128, 800, 3), classes=4, activation='sigmoid')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
    return model, preprocess

def load_data(path, subset = 'train'):
    """Loads data from the drive

    Args:
        path (str): root directory that contains the data to be loaded
        subset (str):  subset of data to be loaded; train or test

    Returns:
          data (ndarray): returns the training or test data.

    Note: training data is returned as 2D array containing both image IDs (only filenames) and its corresponding mask
          information while testing data returns only the image IDs (filenames)
    """
    if subset == 'train':
        df = pd.read_csv(path + 'train.csv')
        df = df[df['EncodedPixels'].notna()]  # remove possible NaNs
        data = {}
        for index, row in df.iterrows():
            imgID = row['ImageId']
            if imgID in data:
                data[imgID]['defectIDs'].append(row['ClassId'] - 1)
                data[imgID]['masks'].append(row['EncodedPixels'])
            else:
                dic = {'defectIDs': [row['ClassId'] - 1], 'masks': [row['EncodedPixels']]}
                data[imgID] = dic

        data = np.array(list(data.items()))

    else:
        _data = [test_file for test_file in listdir(path + 'test_images/')]
        _data = np.array(_data)
        data = np.c_[_data,np.zeros(len(_data))]
    return data

def visualize(path):
    """Visualizes the training data with its masks

    Args:
          path (str): root directory containing the training and testing dataset
    """
    filenames = {}
    data = load_data(path)

    dg = DataGenerator(np.array(data), path, batch_size=16, info=filenames)
    for batch_idx, (X, Y) in enumerate(dg):  # loop batches one by one
        fig = plt.figure(figsize=(16, 25))
        for idx, (img, masks) in enumerate(zip(X, Y)):  # loop of images
            for m in range(4):  # loop different defects
                mask = masks[:, :, m]
                mask = mask2contour(mask, width=2)
                if m == 0:  # yellow
                    img[mask == 1, 0] = 235
                    img[mask == 1, 1] = 235
                elif m == 1:
                    img[mask == 1, 1] = 210  # green
                elif m == 2:
                    img[mask == 1, 2] = 255  # blue
                elif m == 3:  # magenta
                    img[mask == 1, 0] = 255
                    img[mask == 1, 2] = 255
            plt.axis('off')
            fig.add_subplot(8, 2, idx+1)
            plt.imshow(img/255.0)
            plt.title(filenames[16 * batch_idx + idx])
        plt.show()

def train(path, save_file):
    """ Trains the model

    Args:
        path (str): root directory containing the training dataset
        save_file (str): directory to save the trained model for use at a later stage
    """
    filenames = {}
    data = load_data(path)
    model, preprocess = get_model()
    idx = int(0.8 * len(data))
    train_batches = DataGenerator(data[:idx], path, shuffle=True, preprocess=preprocess, info=filenames)
    valid_batches = DataGenerator(data[idx:], path, preprocess=preprocess, info=filenames)
    model.fit_generator(train_batches, validation_data=valid_batches, epochs=5, verbose=1)
    model.save(save_file)

def predict(data_path, model_weights, dataset = 'train'):
    """ Predicts defects on the steel

    Args:
        path (str): root directory containing the testing dataset
        model_weights (str): directory containing the saved model from the train function (save_file)
        dataset (str): choose between train or test data set to run the training on
    """
    model, preprocess = get_model()
    model.load_weights(model_weights)
    data = load_data(data_path, subset=dataset)
    idx = int(0.8 * len(data))
    filenames = {}
    valid_batches = DataGenerator(data[idx:], data_path, preprocess=preprocess, info=filenames, subset=dataset,
                                  batch_size=8)
    preds = model.predict_generator(valid_batches, verbose=1)

    if dataset == 'train':

        for i,batch in enumerate(valid_batches):
            plt.figure(figsize=(20, 72))

            for k in range(8):
                plt.subplot(8,5,5*k+1)
                img = batch[0][k,]
                img = Image.fromarray(img.astype('uint8'))
                img = np.array(img)
                dft = []
                extra = '  has defect(s) '
                for j in range(4):
                    msk = batch[1][k,:,:,j]
                    if np.sum(msk)!=0:
                        dft.append(j+1)
                        extra += ' '+str(j+1)
                    #msk = mask2pad(msk,pad=2)
                    msk = mask2contour(msk,width=3)
                    if j==0: # yellow
                        img[msk==1,0] = 235
                        img[msk==1,1] = 235
                    elif j==1: img[msk==1,1] = 210 # green
                    elif j==2: img[msk==1,2] = 255 # blue
                    elif j==3: # magenta
                        img[msk==1,0] = 255
                        img[msk==1,2] = 255

                if extra=='  has defect(s) ': extra =''
                plt.title('Image originally  ' + extra)
                plt.axis('off')
                plt.imshow(img)

                for defect in range(4):
                    plt.subplot(8,5,5*k+defect+2)
                    msk = preds[8*i+k,:,:,defect]
                    plt.imshow(msk)
                    plt.axis('off')
                    mx = np.round(np.max(msk), 3)
                    if k==0:
                        plt.title('Predicted Mask for Defect ' + str(defect+1))
            plt.tight_layout()
            plt.show()

    if dataset == 'test':
        for i, batch in enumerate(valid_batches):
            plt.figure(figsize=(20, 72))

            for k in range(8):
                plt.subplot(8, 5, 5 * k + 1)
                img = batch[k,]
                img = Image.fromarray(img.astype('uint8'))
                img = np.array(img)
                if k == 0:
                    plt.title('Test Image(s)  ')
                plt.axis('off')
                plt.imshow(img)

                for defect in range(4):
                    plt.subplot(8, 5, 5 * k + defect + 2)  # plt.subplot(16,5,2*k+j)
                    msk = preds[8 * i + k, :, :, defect]
                    plt.imshow(msk)
                    plt.axis('off')
                    mx = np.round(np.max(msk), 3)
                    if k == 0:
                        plt.title('Predicted Mask for Defect ' + str(defect + 1))
            plt.tight_layout()
            plt.show()


def main():
    """Main function of the program the parses the command line arguments and calls appropriate functions

    """
    parser = argparse.ArgumentParser(description='Uses a Conolutional Neural Network for image segmentation to detect '
                                                 'defects in steel')
    parser.add_argument('-dd','--dataset_directory', help= 'Path of training and testing image dataset')
    parser.add_argument('-md','--model_directory', help= 'Path directory to save trained model')
    parser.add_argument('-dt','--dataset', help= 'subset of dataset to be used: train/test', choices= ['train', 'test'])
    parser.add_argument('-v','--visualize', help= 'Visualize Training Dataset', action='store_true')
    parser.add_argument('-t','--train', help= 'Train Model on Dataset', action='store_true')
    parser.add_argument('-p','--predict', help= 'Make Prediction on given dataset',action='store_true')

    args = parser.parse_args()

    if args.train:
        if args.dataset_directory is None or args.model_directory is None:
            print('If you are looking to train a model you need to specify a dataset directory for the training images '
                  'and a model directory to save the model')
            if args.dataset_directory is None:
                print('You have not mentioned a dataset directory. Mention a dataset directory with the -dd or '
                      '--dataset_directory argument')
            if args.model_directory is None:
                print('You have not mentioned a model directory. Mention a model directory with the -md or '
                      '--model_directory argument')
            sys.exit('Program will now exit')
        else:
            train(args.dataset_directory,args.model_directory)
            return

    elif args.visualize:
        if args.dataset_directory is None:
            print('If you want to visualize the dataset you need to specify a dataset directory. You can do so using '
                  'the -dd or --dataset_directory argument')
            sys.exit('Program will now exit')
        else:
            visualize(args.dataset_directory)
            return

    elif args.predict:
        if args.model_directory is None or args.dataset_directory is None or args.dataset is None:
            print('If you are looking to predict defects on a dataset using a model make sure the following three are '
                  'present:'
                  '\n1. model directory using the -md or --model_directory argument that locates the model to be used'
                  '\n2. dataset directory using the -dd or --dataset_directory argument that locates the training or '
                  'testing '
                  'dataset to run predictions on'
                  '\n3. dataset subset using the -dt or --dataset argument which can be only one of two "train" or '
                  '"test" ')
        else:
            predict(args.dataset_directory, args.model_directory,args.dataset)
            return

if __name__ == '__main__':
    main()


