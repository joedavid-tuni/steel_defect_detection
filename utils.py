import numpy as np

def rle2mask(mask_rle, shape=(1600, 256)):
    ''' Converts run-length-encoded mask to numpy array of image dimensions

    Args:
        mask_rle (str): run-length encoding as string formated (start length)
        shape (tuple): (height,width) of array to return

    Returns:
        (ndarray) numpy mask array (1 - mask, 0 - background)
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T # Transposed Needed to accomodate with receiving array dimensions

def mask2contour(mask, width=3):
    ''' Converts mask to contour

        Args:
            mask (ndarray): mask to be converted
            width (int): width of contour

        Returns:
            (ndarray) numpy contour array
        '''
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:, width:], np.zeros((h, width))], axis=1)
    mask2 = np.logical_xor(mask, mask2)
    mask3 = np.concatenate([mask[width:, :], np.zeros((width, w))], axis=0)
    mask3 = np.logical_xor(mask, mask3)
    return np.logical_or(mask2, mask3)
