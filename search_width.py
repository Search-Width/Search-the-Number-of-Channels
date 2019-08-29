import keras
import tensorflow as tf
from keras import Model, optimizers
from keras.preprocessing.image import ImageDataGenerator
from random_eraser import get_random_crop
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.datasets import cifar10
import numpy as np
import copy
import random
from SGDR import SGDRScheduler
from sklearn.model_selection import train_test_split
import os
import time
import resnet_inil
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def wider(conv1, conv2, old_model, add_filter):
    w_conv1, b_conv1 = old_model.get_layer(conv1).get_weights()
    w_conv2, b_conv2 = old_model.get_layer(conv2).get_weights()
    add_num = round(add_filter * w_conv1.shape[3])
    if add_num == 0:
        return old_model
    index = np.random.randint(w_conv1.shape[3], size=add_num)
    factors = np.bincount(index)[index] + 1.
    tmp_w1 = w_conv1[:, :, :, index]
    tmp_b1 = b_conv1[index]
    tmp_w2 = w_conv2[:, :, index, :] / factors.reshape((1, 1, -1, 1))
    noise = np.random.normal(0, 5e-2 * tmp_w2.std(), size=tmp_w2.shape)
    new_w_conv1 = np.concatenate((w_conv1, tmp_w1), axis=3)
    new_w_conv2 = np.concatenate((w_conv2, tmp_w2 + noise), axis=2)
    new_w_conv2[:, :, index, :] = tmp_w2
    new_b_conv1 = np.concatenate((b_conv1, tmp_b1), axis=0)
    model_config = old_model.get_config()
    tmp_name = ''
    shotcut_name = ''
    next_conv = ''

    for one in model_config['layers']:
        if one['config']['name'] == conv1:
            one['config']['filters'] += add_num
            break
        for index2, one_2 in enumerate(model_config['layers']):
            if one_2['config']['name'] == conv1:
                if model_config['layers'][index2 + 1]['class_name'] == 'BatchNormalization':
                    tmp_name = model_config['layers'][index2 + 1]['name']
                elif model_config['layers'][index2 + 1]['class_name'] == 'Add':
                    tmp_name = model_config['layers'][index2 + 2]['name']
                    shotcut_name = model_config['layers'][index2 - 1]['name']
                    next_conv = model_config['layers'][index2 + 7]['name']
                break

    for one in model_config['layers']:
        if one['config']['name'] == shotcut_name:
            one['config']['filters'] += add_num
            break

    a, b, c, d = old_model.get_layer(tmp_name).get_weights()
    tmp_a = a[index]
    tmp_b = b[index]
    tmp_c = c[index]
    tmp_d = d[index]
    new_a = np.concatenate((a, tmp_a), axis=0)
    new_b = np.concatenate((b, tmp_b), axis=0)
    new_c = np.concatenate((c, tmp_c), axis=0)
    new_d = np.concatenate((d, tmp_d), axis=0)

    if shotcut_name != '':
        w_1, b_1 = old_model.get_layer(shotcut_name).get_weights()
        w_2, b_2 = old_model.get_layer(next_conv).get_weights()
        t_w1 = w_1[:, :, :, index]
        t_b1 = b_1[index]
        t_w2 = w_2[:, :, index, :] / factors.reshape((1, 1, -1, 1))
        noise = np.random.normal(0, 5e-2 * t_w2.std(), size=t_w2.shape)
        new_w_1 = np.concatenate((w_1, t_w1), axis=3)
        new_w_2 = np.concatenate((w_2, t_w2 + noise), axis=2)
        new_w_2[:, :, index, :] = t_w2
        new_b_1 = np.concatenate((b_1, t_b1), axis=0)

    new_model = Model.from_config(model_config)
    for one_layer in new_model.layers:
        if one_layer.name == conv1:
            new_model.get_layer(conv1).set_weights([new_w_conv1, new_b_conv1])
        elif one_layer.name == conv2:
            new_model.get_layer(conv2).set_weights([new_w_conv2, b_conv2])
        elif one_layer.name == tmp_name:
            new_model.get_layer(tmp_name).set_weights([new_a, new_b, new_c, new_d])
        elif one_layer.name == shotcut_name:
            new_model.get_layer(shotcut_name).set_weights([new_w_1, new_b_1])
        elif one_layer.name == next_conv:
            new_model.get_layer(next_conv).set_weights([new_w_2, b_2])
        else:
            new_model.get_layer(one_layer.name).set_weights(old_model.get_layer(one_layer.name).get_weights())
    return new_model


def wider_last(conv1, old_model, add_filter):
    w_conv1, b_conv1 = old_model.get_layer(conv1).get_weights()
    add_num = round(add_filter * w_conv1.shape[3])
    if add_num == 0:
        return old_model
    index = np.random.randint(w_conv1.shape[3], size=add_num)
    tmp_w1 = w_conv1[:, :, :, index]
    tmp_b1 = b_conv1[index]
    new_w_conv1 = np.concatenate((w_conv1, tmp_w1), axis=3)
    new_b_conv1 = np.concatenate((b_conv1, tmp_b1), axis=0)
    model_config = old_model.get_config()
    tmp_name = ''
    shotcut_name = ''
    for one in model_config['layers']:
        if one['config']['name'] == conv1:
            one['config']['filters'] += add_num
            break
        for index2, one_2 in enumerate(model_config['layers']):
            if one_2['config']['name'] == conv1:
                if model_config['layers'][index2 + 1]['class_name'] == 'BatchNormalization':
                    tmp_name = model_config['layers'][index2 + 1]['name']
                elif model_config['layers'][index2 + 1]['class_name'] == 'Add':
                    tmp_name = model_config['layers'][index2 + 2]['name']
                    shotcut_name = model_config['layers'][index2 - 1]['name']
                break
    for one in model_config['layers']:
        if one['config']['name'] == shotcut_name:
            one['config']['filters'] += add_num
            break
    a, b, c, d = old_model.get_layer(tmp_name).get_weights()
    tmp_a = a[index]
    tmp_b = b[index]
    tmp_c = c[index]
    tmp_d = d[index]
    new_a = np.concatenate((a, tmp_a), axis=0)
    new_b = np.concatenate((b, tmp_b), axis=0)
    new_c = np.concatenate((c, tmp_c), axis=0)
    new_d = np.concatenate((d, tmp_d), axis=0)

    if shotcut_name != '':
        w_1, b_1 = old_model.get_layer(shotcut_name).get_weights()
        t_w1 = w_1[:, :, :, index]
        t_b1 = b_1[index]
        new_w_1 = np.concatenate((w_1, t_w1), axis=3)
        new_b_1 = np.concatenate((b_1, t_b1), axis=0)

    w, b = old_model.get_layer('dense_1').get_weights()
    zero = np.zeros((add_num, w.shape[1]))
    new_w1 = np.concatenate((w, zero), axis=0)

    new_model = Model.from_config(model_config)
    for one_layer in new_model.layers:
        if one_layer.name == conv1:
            new_model.get_layer(conv1).set_weights([new_w_conv1, new_b_conv1])
        elif one_layer.name == tmp_name:
            new_model.get_layer(tmp_name).set_weights([new_a, new_b, new_c, new_d])
        elif one_layer.name == shotcut_name:
            new_model.get_layer(shotcut_name).set_weights([new_w_1, new_b_1])
        elif one_layer.name == 'dense_1':
            new_model.get_layer('dense_1').set_weights([new_w1, b])
        else:
            new_model.get_layer(one_layer.name).set_weights(old_model.get_layer(one_layer.name).get_weights())
    return new_model

def reload(s, model):
    model.save('./tmp_models/model_reload_tmp' + '.h5')
    s.close()
    keras.backend.clear_session()
    tf.reset_default_graph()
    g = tf.get_default_graph()
    s = tf.InteractiveSession()
    with g.as_default():
        new_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
    return new_model

max_rate = 0.2
conv_num = 16
conv_block = [2, 2, 2, 2]

v1 = []
for i in range(conv_num):
    per_rate = max_rate / 16
    v1.append(per_rate * (i + 1))

v2 = []
count = 1
for i in conv_block:
    for j in range(i * 2):
        per_rate = max_rate / (2 ** (len(conv_block) - 1))
        v2.append(per_rate * count)
    count *= 2

v3 = []
for i in range(conv_num):
    v3.append(max_rate * (((1 + max_rate) ** (i + 1) - 1))/((1 + max_rate) ** conv_num - 1))

v4 = []
for i in range(conv_num):
    per_rate = max_rate / 2
    v4.append(per_rate)

v5 = []
for i in range(conv_num):
    v5.append(max_rate * (((1 + max_rate) ** conv_num - (1 + max_rate) ** (conv_num - i)))/((1 + max_rate) ** conv_num - 1))


(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()
print('train:', len(x_img_train))
print('test:', len(x_img_test))
x_img_train_normalize = x_img_train.astype('float32') / 255.0
y_label_train_OneHot = np_utils.to_categorical(y_label_train)

x_img_train_normalize, x_valid, y_label_train_OneHot, y_valid = train_test_split(x_img_train_normalize, y_label_train_OneHot, test_size=0.2, stratify=y_label_train_OneHot)


datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip = True, preprocessing_function=get_random_crop(crop_shape=[32, 32], padding=4))
datagen_test = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(x_img_train_normalize)
datagen_test.fit(x_valid)


my_model = resnet_inil.ResnetBuilder.build_resnet((3, 32, 32), 10)
plot_model(my_model, to_file='resnet.png', show_shapes=True)
my_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.05, decay=0., momentum=0.9, nesterov=True), metrics=['accuracy'])
schedule = SGDRScheduler(min_lr=0, max_lr=0.05, steps_per_epoch=np.ceil(x_img_train_normalize.shape[0] / 128), lr_decay=1, cycle_length=1, mult_factor=2)
record = my_model.fit_generator(datagen.flow(x_img_train_normalize, y_label_train_OneHot, batch_size=128), epochs=31, verbose=1, callbacks=[schedule])
scores2 = my_model.evaluate_generator(datagen_test.flow(x_valid, y_valid), verbose=0)
print("pre-training round 2(SGDR):  Train: " + str(record.history['acc'][-1]) + "  Valid: " + str(scores2[1]))
print("Number of parameters: " + str(round(my_model.count_params() / 1000000, 2)) + "M")


global sess
pop_size = 12
my_model.save('./tmp_models/init_model.h5')
best_score_list = [copy.deepcopy(scores2[1]) for one in range(pop_size)]
train_acc_list = [copy.deepcopy(record.history['acc'][-1]) for one2 in range(pop_size)]
ancestry_list = []
mutate_list = []
for a in range(pop_size):
    ancestry_list.append(str(a))
    if a > 0:
        sess.close()
        keras.backend.clear_session()
    else:
        keras.backend.clear_session()
    tf.reset_default_graph()
    g = tf.get_default_graph()
    sess = tf.InteractiveSession()
    with g.as_default():
        my_model = keras.models.load_model('./tmp_models/init_model.h5')
    conv_list = [2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24]
    random_num = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print('--------' + str(random_num)+ '--------')
    if random_num == 1:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v1[k])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v1[k])
    elif random_num == 2:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v1[0])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v1[conv_num - k - 1])
    elif random_num == 3:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v2[k])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v2[k])
    elif random_num == 4:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v2[0])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v2[conv_num - k - 1])
    elif random_num == 5:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v3[k])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v3[k])
    elif random_num == 6:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v3[0])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v3[conv_num - k - 1])
    elif random_num == 7:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v4[k])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v4[k])
    elif random_num == 8:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v5[k])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v5[k])
    else:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v5[0])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v5[conv_num - k - 1])

    my_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.05, decay=0., momentum=0.9, nesterov=True), metrics=['accuracy'])
    schedule = SGDRScheduler(min_lr=0, max_lr=0.05, steps_per_epoch=np.ceil(x_img_train_normalize.shape[0] / 128),lr_decay=1, cycle_length=1, mult_factor=2)
    record = my_model.fit_generator(datagen.flow(x_img_train_normalize, y_label_train_OneHot, batch_size=128), epochs=15,verbose=1, callbacks=[schedule])
    scores2 = my_model.evaluate_generator(datagen_test.flow(x_valid, y_valid), verbose=0)
    best_score_list[a] = copy.deepcopy(scores2[1])
    train_acc_list[a] = copy.deepcopy(record.history['acc'][-1])
    my_model.save('./tmp_models/model_' + str(a) + '.h5')
    print("----------- model_num:  " + str(a) + " -----------")
    print("----------- Valid:  " + str(scores2[1]) + " -----------")
print(best_score_list)
best_score = max(best_score_list)
best_index = best_score_list.index(best_score)
print("----------- best_index:  " + str(best_index) + " -----------")
print("----------- best_score:  " + str(best_score) + " -----------")
my_model = keras.models.load_model('./tmp_models/model_' + str(best_index) + '.h5')
plot_model(my_model, to_file='func_preserve.png', show_shapes=True)
participate_list = copy.deepcopy(best_score_list)


tournament_size = 3
round_num = 1
while best_score < 0.97:
    print("+++++++++++++++++++++++++++++++++++++++")
    print("----------- round: " + str(round_num) + " -----------")
    round_num += 1
    tournament_pop = random.sample(participate_list, tournament_size)
    tournament_best = max(tournament_pop)
    tournament_best_index = best_score_list.index(tournament_best)
    pop_size += 1

    best_score_list.append(copy.deepcopy(best_score_list[tournament_best_index]))
    participate_list.append(copy.deepcopy(best_score_list[tournament_best_index]))
    train_acc_list.append(copy.deepcopy(train_acc_list[tournament_best_index]))

    sess.close()
    keras.backend.clear_session()
    tf.reset_default_graph()
    g = tf.get_default_graph()
    sess = tf.InteractiveSession()
    with g.as_default():
        my_model = keras.models.load_model('./tmp_models/model_' + str(tournament_best_index) + '.h5')
        print('Load model: ' + './tmp_models/model_' + str(tournament_best_index) + '.h5')
    conv_list = [2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24]
    random_num = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print('--------' + str(random_num) + '--------')
    if random_num == 1:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v1[k])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v1[k])
    elif random_num == 2:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v1[0])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v1[conv_num - k - 1])
    elif random_num == 3:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v2[k])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v2[k])
    elif random_num == 4:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v2[0])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v2[conv_num - k - 1])
    elif random_num == 5:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v3[k])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v3[k])
    elif random_num == 6:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v3[0])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v3[conv_num - k - 1])
    elif random_num == 7:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v4[k])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v4[k])
    elif random_num == 8:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v5[k])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v5[k])
    else:
        for k in range(len(conv_list)):
            my_model.save('./tmp_models/model_reload_tmp' + '.h5')
            sess.close()
            keras.backend.clear_session()
            tf.reset_default_graph()
            g = tf.get_default_graph()
            sess = tf.InteractiveSession()
            with g.as_default():
                my_model = keras.models.load_model('./tmp_models/model_reload_tmp' + '.h5')
            if k == len(conv_list) - 1:
                my_model = wider_last('conv2d_' + str(conv_list[k]), my_model, v5[0])
            else:
                my_model = wider('conv2d_' + str(conv_list[k]), 'conv2d_' + str(conv_list[k + 1]), my_model, v5[conv_num - k - 1])

    my_model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.05, decay=0., momentum=0.9, nesterov=True), metrics=['accuracy'])
    schedule = SGDRScheduler(min_lr=0, max_lr=0.05, steps_per_epoch=np.ceil(x_img_train_normalize.shape[0] / 128), lr_decay=1, cycle_length=1, mult_factor=2)

    new_record = my_model.fit_generator(datagen.flow(x_img_train_normalize, y_label_train_OneHot, batch_size=128), epochs=15, verbose=1, callbacks=[schedule])
    new_score = my_model.evaluate_generator(datagen_test.flow(x_valid, y_valid), verbose=0)
    best_score_list[-1] = copy.deepcopy(new_score[1])
    participate_list[-1] = copy.deepcopy(new_score[1])
    train_acc_list[-1] = copy.deepcopy(new_record.history['acc'][-1])
    my_model.save('./tmp_models/model_' + str(pop_size - 1) + '.h5')

    print(best_score_list)
    if round_num > 8:
        random_op = random.choice([1])
        print('random_op: ' + str(random_op))
        if random_op == 0:
            participate_list.pop(0)
        else:
            participate_list.remove(min(participate_list))
    print(participate_list)
    best_score = max(best_score_list)
    best_index = best_score_list.index(best_score)
    print("----------- best_index:  " + str(best_index) + " -----------")
    print("----------- best_score:  " + str(best_score) + " -----------")
    my_model = keras.models.load_model('./tmp_models/model_' + str(best_index) + '.h5')
    print("----------- best_model_parameters: " + str(round(my_model.count_params() / 1000000, 2)) + "M -----------")
    plot_model(my_model, to_file='func_preserve.png', show_shapes=True)
    my_model.save('best_model.h5')

    output = open('search_log.txt', 'a+')
    output.write(str(round_num - 1))
    output.write(":   Train:")
    output.write(str(train_acc_list[best_index]))
    output.write("  Valid:")
    output.write(str(best_score))
    output.write(" (write-in time: ")
    output.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 43200))))
    output.write(")")
    output.write("------")
    output.write(str(best_score_list))
    output.write('\n')
    output.close()

