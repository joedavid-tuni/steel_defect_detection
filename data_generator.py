import numpy as np
import keras
from utils import rle2mask
import matplotlib.pyplot as plt


class DataGenerator(keras.utils.Sequence):
    """ Generates real-time data feed to model in batches

    """

    def __init__(self, data, path, img_size=(128, 800), batch_size=16, subset="train", shuffle=False,
                 preprocess=None, info={}):
        """ Initializes the DataGenerator Object

        Args:
            data (numpy.ndarray): All of data from which to create batches
            shuffle (bool) : whether to shuffle the order in which data is fed to the model
            subset (str): which subset of dataset to use; Trainining or testing
            batch_size (int): size of batch that arrives as data feed
            preprocess : pre-processing BACKBONE
            info (str) : image IDs generated at each pass
            path (str) : root directory containing the training and testing dataset
            img_size (tuple): size of the image to be provided in the data feed

        """


        super().__init__()
        self.data = data
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info
        self.path = path
        self.img_size = img_size

        if self.subset == "train":
            self.data_path = path + 'train_images/'
        elif self.subset == "test":
            self.data_path = path + 'test_images/'
        self.on_epoch_end()

    def __len__(self):
        """Calculates number of batch in the Sequence.

        Returns:
            (int): The number of batches in the Sequence.
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        """ A method called at the end of every epoch that shuffles the data if parameter shuffle = True.

        """
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generates one batch of data at position 'index'.

        Note: index=0 for batch 1, 2 for batch 2 and so on..

        Arguments:
            index (int): position of the batch in the Sequence.
        Returns:
            A batch.
            X (ndarray): array containing the image
                        4D array of size: batch_size x img_size[0] x img_size[1] x 3 (RGB)
            Y (ndarray): array containing the masks for corresponding images in X
                        4D array of size: batch_size x img_size[0] x img_size[1] x 4 (number of defect classes)

        Note: If subset =' train', both the images along with its masks is returned. This is essentially the information
        contained in the train.csv file. If subset = 'test', only the images in the test_images folder is returned
        """
        # (batch size, image height, image width, number of channels (RGB=3))
        X = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 3), dtype=np.float32)

        # (batch size, image height, image width, number of (defect) classes = 4 (one hot coded) )
        Y = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 4), dtype=np.int8)

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        for idx, (imgID, masks) in enumerate(self.data[indexes]):
            self.info[index * self.batch_size + idx] = imgID
            X[idx, ] = plt.imread(self.data_path + imgID)[::2, ::2]
            if self.subset == 'train':
                defectsIDs = masks['defectIDs']
                masks = masks['masks']
                for m in range(len(defectsIDs)):
                    Y[idx, :, :, defectsIDs[m]] = rle2mask(masks[m])[::2, ::2]
        if self.preprocess != None: X = self.preprocess(X)
        if self.subset == 'train':
            return X, Y
        else:
            return X