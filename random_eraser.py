import numpy as np


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

def get_cutout(n_holes=1, length=16):
    def cutout(input_img):
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
    return cutout

def get_random_crop(crop_shape=[32, 32], padding=4):
    def random_crop(input_img):
        npad = ((padding, padding), (padding, padding), (0, 0))
        input_img = np.lib.pad(input_img, pad_width=npad, mode='constant', constant_values=0)
        nh = np.random.randint(0, 32 + 2 * padding - crop_shape[0])
        nw = np.random.randint(0, 32 + 2 * padding - crop_shape[1])
        input_img = input_img[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        return input_img
    return random_crop

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