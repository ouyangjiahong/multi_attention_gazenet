
"""Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
import scipy.misc as sci
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial

from keras import backend as K
from keras.utils.data_utils import Sequence

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.
    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x

def load_img(path, grayscale=False, target_size=None, crop=True,
             interpolation='nearest'):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    width = img.size[0]
    height = img.size[1]
    if crop:
        img = img.crop(
            (
                width - height,
                0,
                width,
                height
            )
        )
        # print(img.size)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))

            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img, width, height


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


class GazeDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width, if < 1, or pixels if >= 1.
        height_shift_range: fraction of total height, if < 1, or pixels if >= 1.
        brightness_range: the range of brightness to apply
        shear_range: shear intensity (shear angle in degrees).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channel.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
            Points outside the boundaries of the input are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: fraction of images reserved for validation (strictly between 0 and 1).
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 validation_split=0.0):
        if data_format is None:
            data_format = K.image_data_format()
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2
        if validation_split and not 0 < validation_split < 1:
            raise ValueError('`validation_split` must be strictly between 0 and 1. '
                             ' Received arg: ', validation_split)
        self._validation_split = validation_split

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)
        if zca_whitening:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, which overrides '
                              'setting of `featurewise_center`.')
            if featurewise_std_normalization:
                self.featurewise_std_normalization = False
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening` '
                              'which overrides setting of'
                              '`featurewise_std_normalization`.')
        if featurewise_std_normalization:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, '
                              'which overrides setting of '
                              '`featurewise_center`.')
        if samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`samplewise_std_normalization`, '
                              'which overrides setting of '
                              '`samplewise_center`.')

    # def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
    #          save_to_dir=None, save_prefix='', save_format='png', subset=None):
    #     return NumpyArrayIterator(
    #         x, y, self,
    #         batch_size=batch_size,
    #         shuffle=shuffle,
    #         seed=seed,
    #         data_format=self.data_format,
    #         save_to_dir=save_to_dir,
    #         save_prefix=save_prefix,
    #         save_format=save_format,
    #         subset=subset)

    def flow_from_directory(self, directory,
                            time_steps=500,
                            time_skip=1,
                            crop=True,
                            gaussian_std=0.01,
                            target_size=(256, 256), color_mode='rgb',
                            crop_with_gaze=False, crop_with_gaze_size=128,
                            classes=None, class_mode='sequence',
                            batch_size=1, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        print("get into flow from directory")
        return DirectoryIterator(
            directory, self,
            time_steps=time_steps,
            time_skip=time_skip,
            crop=crop,
            gaussian_std=gaussian_std,
            target_size=target_size, color_mode=color_mode,
            crop_with_gaze=crop_with_gaze, crop_with_gaze_size=crop_with_gaze_size,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.
        # Arguments
            x: batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        if self.rescale:
            x *= self.rescale
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, keepdims=True) + K.epsilon())

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + K.epsilon())
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def random_transform(self, x, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.deg2rad(np.random.uniform(-self.rotation_range, self.rotation_range))
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if self.height_shift_range < 1:
                tx *= x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            if self.width_shift_range < 1:
                ty *= x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range))
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)

        if self.brightness_range is not None:
            x = random_brightness(x, self.brightness_range)

        return x

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Fits internal statistics to some sample data.
        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' + self.data_format + '" '
                '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
            self.principal_components = (u * s_inv).dot(u.T)


class Iterator(Sequence):
    """Base class for image data iterators.
    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        print('orig')
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.reset()
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Count files with extension in `white_list_formats` contained in directory.
    # Arguments
        directory: absolute path to the directory
            containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        follow_links: boolean.
    # Yields
        tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            for extension in white_list_formats:
                if fname.lower().endswith('.tiff'):
                    warnings.warn('Using \'.tiff\' files with multiple bands will cause distortion. '
                                  'Please verify your output.')
                if fname.lower().endswith('.' + extension):
                    yield root, fname

def _iter_valid_interaction(directory, white_list_formats, time_steps, time_skip, follow_links):
    # Return List of Valid Interactions
    valid_interactions = []
    for subdir in sorted(os.listdir(directory)):
        interactions = [os.path.join(directory, subdir, dirname) for dirname in os.listdir(os.path.join(directory, subdir))]
        for inter in interactions:
            if len(list(_iter_valid_files(inter, white_list_formats, follow_links))) > time_steps*time_skip:
                valid_interactions.append(inter)
    return valid_interactions

def _iter_valid_interaction_in_directory(directory, white_list_formats, time_steps, time_skip, follow_links):
    # Return List of Valid Interaction in the Directory (Does not go into Subdirectory)
    valid_interactions = []
    interactions = [os.path.join(directory, dirname) for dirname in os.listdir(os.path.join(directory))]
    # print(interactions)
    for inter in interactions:
        # if len(list(_iter_valid_files(inter, white_list_formats, follow_links))) > time_steps*time_skip:
        valid_interactions.append(inter)
    return valid_interactions

def _count_valid_interaction_number_in_directory(directory, white_list_formats, time_steps, time_skip, split, follow_links):
    # Count Number of Valid Interactions
    # print(directory)
    num_interations = len(_iter_valid_interaction_in_directory(directory, white_list_formats, time_steps, time_skip, follow_links))
    if split:
        start, stop = int(split[0] * num_interations), int(split[1] * num_interations)
    else:
        start, stop = 0, num_interations
    return stop - start


def _count_valid_files_in_directory(directory, white_list_formats, split, follow_links):
    """Count files with extension in `white_list_formats` contained in directory.
    # Arguments
        directory: absolute path to the directory
            containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        follow_links: boolean.
    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """
    num_files = len(list(_iter_valid_files(directory, white_list_formats, follow_links)))
    if split:
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
    else:
        start, stop = 0, num_files
    return stop - start


def _list_valid_filenames_in_directory(directory, white_list_formats, split,
                                       class_indices, follow_links):
    """List paths of files in `subdir` with extensions in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean.
    # Returns
        classes: a list of class indices
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
    """
    dirname = os.path.basename(directory)
    if split:
        num_files = len(list(_iter_valid_files(directory, white_list_formats, follow_links)))
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = list(_iter_valid_files(directory, white_list_formats, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_files(directory, white_list_formats, follow_links)

    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames

def _list_valid_interactions_in_directory(directory, white_list_formats, time_steps, time_skip, split,
                                       class_indices, follow_links):
    # List directory names with valid interactions in the 'class' directory

    dirname = os.path.basename(directory)
    if split:
        num_interactions = len(_iter_valid_interaction_in_directory(directory, white_list_formats, time_steps, time_skip, follow_links))
        start, stop = int(split[0] * num_interactions), int(split[1] * num_interactions)
        valid_files = list(_iter_valid_interaction_in_directory(directory, white_list_formats, time_steps, time_skip, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_interaction_in_directory(directory, white_list_formats, time_steps, time_skip, follow_links)

    classes = []
    internames = []
    for absolute_path in valid_files:
        classes.append(class_indices[dirname])
        # absolute_path = os.path.join(root, fname)
        # relative_path = os.path.join(dirname, os.path.relpath(absolute_path, directory))
        internames.append(absolute_path)

    return classes, internames


def load_label_sequence(interaction_path, label_file_name='label.npy'):
    # Load the npy file that contains label data
    label_sequence = np.load(os.path.join(interaction_path, label_file_name))
    return label_sequence

def load_gaze_sequence(interaction_path, gaze_file_name='gaze.npy'):
    # Load the npy file that contains gaze data
    gaze_sequence = np.load(os.path.join(interaction_path, gaze_file_name))
    return gaze_sequence

def modify_gaze_sequence(gaze_seq, ori_width=640, ori_height=360, crop=True, target_size=None):
    # print(gaze_seq.shape)
    gaze_seq[:, 1] *= ori_width
    gaze_seq[:, 2] *= ori_height
    delta = ori_width - ori_height
    if crop == True:
        # print("crop gaze")
        gaze_seq[:, 1] -= delta
        if target_size != None or target_size != (ori_height, ori_height):
            gaze_seq[:, 1] *= target_size[1] / float(ori_width)
            gaze_seq[:, 2] *= target_size[0] / float(ori_height)
    gaze_seq[:, 1] = gaze_seq[:, 1].astype(int)
    gaze_seq[:, 2] = gaze_seq[:, 2].astype(int)
    return gaze_seq

def load_interaction_sequence(interaction_path, white_list_formats, grayscale=False, time_steps=500, time_skip=1, target_size=None, crop=True, interpolation='nearest'):
    img_sequence = []

    fnames = list(_iter_valid_files(interaction_path, white_list_formats, False))
    gaze_sequence = load_gaze_sequence(interaction_path)
    label_sequence = load_label_sequence(interaction_path)

    if len(fnames) > time_steps*time_skip:
        start = np.random.choice(len(fnames) - time_steps * time_skip, size=1)[0]
        end = time_steps*time_skip
    else:
        end = len(fnames)
        start = 0

    for i in xrange(start, start+end, time_skip):
        root, fname = fnames[i]
        img_path = os.path.join(root, fname)
        # print (img_path)
        img, width, height = load_img(img_path, grayscale=grayscale, target_size=target_size, interpolation=interpolation, crop=crop)
        x = img_to_array(img, data_format=None)
        img_sequence.append(x)
    # print(interaction_path)
    # print("gaze")
    # print(gaze_sequence[start:start+end:time_skip, :].shape)
    # print("label")
    # print(label_sequence[start:start+end:time_skip, :].shape)
    # print("image")
    # print(len(img_sequence))
    if label_sequence[start:start+end:time_skip, :].shape[0] != len(img_sequence):
        img_sequence.pop()
    gaze_sequence = modify_gaze_sequence(gaze_sequence, ori_width=width, ori_height=height, crop=crop, target_size=target_size)
    assert len(img_sequence) <= time_steps
    # print('label sequence shape')
    # print(label_sequence.shape)
    return np.array(img_sequence), gaze_sequence[start:start+end:time_skip, :], label_sequence[start:start+end:time_skip, :]

class DirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        crop: Crop the image so its a square, retaining information on the right side of image
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self, directory, image_data_generator,
                 time_skip=1,
                 time_steps=500,
                 crop=True,
                 gaussian_std=0.01,
                 target_size=(360, 360), color_mode='rgb',
                 crop_with_gaze=False, crop_with_gaze_size=128,
                 classes=None, class_mode='sequence',
                 batch_size=1, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest'):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', 'sequence', 'sequence_pytorch', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input", "sequence"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.time_steps = time_steps
        self.time_skip = time_skip
        self.crop = crop
        self.gaussian_std = gaussian_std
        self.crop_with_gaze = crop_with_gaze
        self.crop_with_gaze_size = crop_with_gaze_size

        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError('Invalid subset name: ', subset,
                                 '; expected "training" or "validation"')
        else:
            split = None
        self.subset = subset

        self.white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff'}

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_interaction_number_in_directory,
                                   time_steps=time_steps,
                                   time_skip=self.time_skip,
                                   white_list_formats=self.white_list_formats,
                                   follow_links=follow_links,
                                   split=split)
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))

        print('Found %d Interactions belonging to %d classes.' % (self.samples, self.num_classes))

        # second, build an index of the images in the different class subfolders
        results = []
        self.dataset_dict = {}
        self.curr_dataset = []
        # print(self.samples)
        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(pool.apply_async(_list_valid_interactions_in_directory,
                                            (dirpath, self.white_list_formats, self.time_steps, self.time_skip, split,
                                             self.class_indices, follow_links)))
        self.min_class_size = float('inf')
        self.internames = []
        for res in results:
            classes, internames = res.get()
            print(classes)
            self.dataset_dict[classes[0]] = internames
            # print(self.classes)
            # print(classes)

            class_size = len(classes)
            if class_size < self.min_class_size:
                self.min_class_size = class_size
            self.classes[i:i + len(classes)] = classes
            self.internames += internames
            i += len(classes)

        self.data_set_size = self.min_class_size * self.num_classes

        pool.close()
        pool.join()
        super(DirectoryIterator, self).__init__(self.data_set_size, batch_size, shuffle, seed)
        # self.internames = None
        print('There are %d Interactions per class for to %d classes.' % (self.min_class_size, self.num_classes))

    def reset(self):
        # print('newfile')
        self.internames = []
        for d in self.dataset_dict:
            self.internames += list(np.random.choice(self.dataset_dict[d], size=(self.min_class_size), replace=False))

        self.batch_index = 0

    def _crop_with_gaze(self, images, gazes):
        batch_size = self.batch_size
        time_steps = self.time_steps
        img_size = self.crop_with_gaze_size
        num_channel = 3
        # print("get into _crop_with_gaze")
        # print((batch_size,time_steps,img_size,img_size,num_channel))
        img_seq = np.zeros((batch_size,time_steps,img_size,img_size,num_channel), dtype=int)
        for i in range(batch_size):
            for j in range(time_steps):
                if (gazes[i,j,1] >= images.shape[3] or gazes[i,j,1] < 0 or gazes[i,j,2] >= images.shape[2] or gazes[i,j,2] < 0) and j == 0 :
                    gazes[i,j,1] = 0
                    gazes[i,j,2] = 0
                if (gazes[i,j,1] >= images.shape[3] or gazes[i,j,1] < 0 or gazes[i,j,2] >= images.shape[2] or gazes[i,j,2] < 0) and j != 0 :
                    gazes[i,j,1] = gazes[i,j-1,1]
                    gazes[i,j,2] = gazes[i,j-1,2]
                x = gazes[i, j, 1]
                y = gazes[i, j, 2]

                right_bound = int(min(x+(img_size/2),images.shape[3]))
                left_bound = int(max(0,x-(img_size/2)))
                up_bound = int(max(0,y-(img_size/2)))
                down_bound = int(min(images.shape[2],y+(img_size/2)))

                tmp= images[i, j, up_bound:down_bound, left_bound:right_bound, :]
                if tmp.shape != (img_size, img_size):
                    img_seq[i, j, :, :, :] = sci.imresize(tmp, (img_size, img_size,3))
                else:
                    img_seq[i, j, :, :, :] = tmp
        # print("finished crop with gaze")
        return img_seq

    def _get_batches_of_transformed_samples(self, index_array):
        # images_x = np.zeros((len(index_array),) + (self.time_steps, ) + self.image_shape, dtype=K.floatx())
        # gaze_x = np.zeros((len(index_array),) + (self.time_steps, ) + (3,), dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        # print(images_x.shape)
        # print(gaze_x.shape)
        # print("idnex array")
        # print(index_array)
        for i, j in enumerate(index_array):
            intername = self.internames[j]
            print(intername)
            img_sequence, gaze_sequence, label_sequence = load_interaction_sequence(intername, self.white_list_formats,
                                                                    grayscale=False, time_steps=self.time_steps,
                                                                    time_skip=self.time_skip, target_size=self.target_size,
                                                                    crop=self.crop, interpolation='nearest')
            if self.gaussian_std:
                # print(gaze_sequence.shape)
                gaussian = np.random.normal(0, self.gaussian_std, gaze_sequence.shape)
                # print(gaussian.shape)
                gaussian[:,0] = 0
                gaze_sequence = gaze_sequence + gaussian
            images_x = img_sequence
            gaze_x = gaze_sequence

        if self.crop_with_gaze == True:
            # print("next, before calling crop with gaze")
            images_x = self._crop_with_gaze(images_x, gaze_x)
            # print(images_x.shape)


        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        # print("before return in get batches")
        if self.class_mode == 'input':
            batch_y = images_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(images_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'sequence':
            # batch_y = label_sequence
            batch_y = np.zeros((len(label_sequence), self.num_classes+1), dtype=K.floatx())
            for i, label in enumerate(label_sequence):
                batch_y[i, int(label[0])] = 1
        elif self.class_mode == 'sequence_pytorch':
            batch_y = np.squeeze(label_sequence, axis=1)
        else:
            return images_x
        # print("before return")
        if self.crop_with_gaze == True:
            return images_x, batch_y
        else:
            return [images_x, gaze_x], batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        # print("get into next")
        return self._get_batches_of_transformed_samples(index_array)
