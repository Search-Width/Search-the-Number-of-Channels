import numpy as np

def get_cutout_crop(crop_shape=[32, 32], padding=4, n_holes=1, length=16):
    def cutout_crop(input_img):
        npad = ((padding, padding), (padding, padding), (0, 0))
        input_img = np.lib.pad(input_img, pad_width=npad, mode='constant', constant_values=0)
        nh = np.random.randint(0, 32 + 2 * padding - crop_shape[0])
        nw = np.random.randint(0, 32 + 2 * padding - crop_shape[1])
        input_img = input_img[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        unmodified = np.random.choice([1, 2])
        if unmodified == 1:
            return input_img
        h, w, c = input_img.shape
        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            input_img[y1: y2, x1: x2] = 0.
        return input_img
    return cutout_crop