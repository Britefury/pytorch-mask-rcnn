import os
import sys
import pickle
import numpy as np
import tqdm
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from matplotlib import pyplot as plt
from PIL import Image
from skimage.util import img_as_float, img_as_ubyte
from skimage.color import grey2rgb

if sys.version_info[0] == 2:  # pragma: no cover
    from ConfigParser import RawConfigParser
else:
    from configparser import RawConfigParser



_CONFIG_PATH = './dsbowl2018.cfg'

_config__ = None

def get_config():  # pragma: no cover
    global _config__
    if _config__ is None:
        if os.path.exists(_CONFIG_PATH):
            try:
                _config__ = RawConfigParser()
                _config__.read(_CONFIG_PATH)
            except Exception as e:
                print('WARNING: error {} trying to open config '
                      'file from {}'.format(e, _CONFIG_PATH))
                _config__ = RawConfigParser()
        else:
            _config__ = RawConfigParser()
    return _config__


def get_config_dir(name):  # pragma: no cover
    dir_path = get_config().get('paths', name)
    if not os.path.exists(dir_path):
        raise RuntimeError(
            'kaggle-cellnucleus.settings: the directory path {} does not exist'.format(dir_path))
    return dir_path


def _rgb_path(ds_dir, sample_name):
    return os.path.join(ds_dir, sample_name, 'images', '{}.png'.format(sample_name))

def _labels_path(ds_dir, sample_name):
    return os.path.join(ds_dir, sample_name, 'labels.png')


class CellNucleusSegDataset (object):
    class ImageAccessor (object):
        def __init__(self, paths, image_load_fn):
            self.paths = paths
            self.image_load_fn = image_load_fn

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, item):
            if isinstance(item, int):
                return self.image_load_fn(self.paths[item])
            else:
                xs = []
                if isinstance(item, slice):
                    indices = range(*item.indices(len(self)))
                elif isinstance(item, np.ndarray):
                    indices = item
                else:
                    raise TypeError('item should be an int/long, a slice or an array, not a {}'.format(
                        type(item)
                    ))
                for i in indices:
                    img = self.image_load_fn(self.paths[i])
                    xs.append(img)
                return xs


    def __init__(self, range01, rgb_order,
                 root_dir, sample_names, rgb_paths, label_paths,
                 sample_classes=None, dummy=False, subset_indices=None, convex_hulls_required=False):
        self.range01 = range01
        self.rgb_order = rgb_order
        self.dummy = dummy

        self.root_dir = root_dir
        self.names = sample_names
        self.rgb_paths = rgb_paths
        self.label_paths = label_paths
        self.cls = np.array(sample_classes) if sample_classes is not None else None


        if subset_indices is not None:
            self.sample_names = [self.sample_names[n] for n in subset_indices]
            self.rgb_paths = [self.rgb_paths[n] for n in subset_indices]
            if self.label_paths is not None:
                self.label_paths = [self.label_paths[n] for n in subset_indices]
            if self.cls is not None:
                self.cls = self.cls[subset_indices]

        self.X = self.ImageAccessor(self.rgb_paths, self.load_rgb_image)
        self.y = None

        if label_paths is not None:
            self.y = self.ImageAccessor(self.label_paths, self.load_label_image)

            if convex_hulls_required:
                # Label convex hulls
                convex_hulls_path = os.path.join(root_dir, 'convex_hulls.pkl')
                if os.path.exists(convex_hulls_path):
                    with open(convex_hulls_path, 'rb') as f:
                        self.convex_hulls = pickle.load(f)
                else:
                    self.convex_hulls = []
                    print('Generating convex hulls...')
                    error = False
                    for sample_i in tqdm.tqdm(range(len(self.y))):
                        labels = self.y[sample_i]

                        sample_convex_hulls = [None]

                        for label_i in range(1, labels.max() + 1):
                            mask = labels == label_i
                            outline_y, outline_x = np.where(mask)
                            outline_points = np.append(outline_y[:, None], outline_x[:, None], axis=1)
                            if len(outline_points) > 0:
                                try:
                                    hull = ConvexHull(outline_points)
                                except QhullError:
                                    print(self.label_paths[sample_i], label_i, outline_points.shape)
                                    plt.figure(figsize=(12,12))
                                    plt.imshow(mask, cmap='gray')
                                    plt.show()
                                    error=True
                                else:
                                    ch_points = outline_points[hull.vertices, :]
                            else:
                                ch_points = np.zeros((0, 2))

                            sample_convex_hulls.append(ch_points)

                        assert len(sample_convex_hulls) == labels.max() + 1

                        self.convex_hulls.append(sample_convex_hulls)
                    if error:
                        raise RuntimeError

                    with open(convex_hulls_path, 'wb') as f:
                        pickle.dump(self.convex_hulls, f)
            else:
                self.convex_hulls = None



    def get_image_size(self, index):
        path = self.rgb_paths[index]
        img = Image.open(path)
        return img.size[1], img.size[0]

    def load_rgb_image(self, path):
        if self.dummy:
            return np.random.randint(0, 256, size=(256, 256, 3)).astype(np.uint8)
        else:
            img = Image.open(path)
            img = np.array(img)
            if img.ndim == 2:
                img = grey2rgb(img)
            else:
                img = img[:, :, :3]
            if not self.rgb_order:
                img = img[:, :, ::-1]
            if self.range01:
                img = img_as_float(img)
            return img


    def load_label_image(self, path):
        if self.dummy:
            return np.random.uniform(0, 1.0, size=(256, 256)).astype(np.float32)
        else:
            img = Image.open(path)
            return np.array(img)


class Stage1TrainSegDataset (CellNucleusSegDataset):
    def __init__(self, range01=True, rgb_order=True, dummy=False, subset_indices=None, exclude_errors=True):
        root_dir = os.path.join(get_config_dir('data_root'), 'stage1_train')
        manifest_path = os.path.join(root_dir, 'exp_cls.pkl')

        if exclude_errors:
            # An unlabelled image
            exclude_names = ['7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80']
        else:
            exclude_names = []

        with open(manifest_path, 'rb') as f_manifest:
            manifest = pickle.load(f_manifest)
            sample_names = manifest['sample_names']
            sample_classes = manifest['sample_classes']

        for excl in exclude_names:
            i = sample_names.index(excl)
            del sample_classes[i]
            del sample_names[i]

        rgb_paths = []
        label_paths = []
        for sample_name in sample_names:
            rgb_paths.append(_rgb_path(root_dir, sample_name))
            label_paths.append(_labels_path(root_dir, sample_name))

        super(Stage1TrainSegDataset, self).__init__(range01, rgb_order, root_dir, sample_names,
                                                    rgb_paths, label_paths,
                                                    sample_classes=sample_classes, dummy=dummy, subset_indices=subset_indices)

        self.labels_csv_path = os.path.join(root_dir, 'stage1_train_labels.csv')



class Stage1TestSegDataset (CellNucleusSegDataset):
    def __init__(self, range01=True, rgb_order=True, dummy=False, subset_indices=None):
        root_dir = os.path.join(get_config_dir('data_root'), 'stage1_test')

        filenames = os.listdir(root_dir)
        sample_names = []
        for filename in filenames:
            p = os.path.join(root_dir, filename)
            if os.path.isdir(p):
                sample_names.append(filename)

        rgb_paths = []
        label_paths = []
        labels_present = True
        for sample_name in sample_names:
            rgb_paths.append(_rgb_path(root_dir, sample_name))
            label_paths.append(_labels_path(root_dir, sample_name))
            if not os.path.exists(label_paths[-1]):
                labels_present = False

        if not labels_present:
            label_paths = None

        super(Stage1TestSegDataset, self).__init__(range01, rgb_order, root_dir, sample_names, rgb_paths, label_paths,
                                                   None, dummy=dummy, subset_indices=subset_indices)


class Stage1TrainTestSegDataset (CellNucleusSegDataset):
    def __init__(self, range01=True, rgb_order=True, dummy=False, subset_indices=None):
        root_dir = os.path.join(get_config_dir('data_root'), 'stage1_traintest')

        filenames = os.listdir(root_dir)
        sample_names = []
        for filename in filenames:
            p = os.path.join(root_dir, filename)
            if os.path.isdir(p):
                sample_names.append(filename)

        rgb_paths = []
        label_paths = []
        labels_present = True
        for sample_name in sample_names:
            rgb_paths.append(_rgb_path(root_dir, sample_name))
            label_paths.append(_labels_path(root_dir, sample_name))
            if not os.path.exists(label_paths[-1]):
                labels_present = False

        if not labels_present:
            label_paths = None

        super(Stage1TrainTestSegDataset, self).__init__(range01, rgb_order, root_dir, sample_names, rgb_paths, label_paths,
                                                   None, dummy=dummy, subset_indices=subset_indices)


class Stage2TestSegDataset (CellNucleusSegDataset):
    def __init__(self, range01=True, rgb_order=True, dummy=False, subset_indices=None):
        root_dir = os.path.join(get_config_dir('data_root'), 'stage2_test')

        filenames = os.listdir(root_dir)
        sample_names = []
        for filename in filenames:
            p = os.path.join(root_dir, filename)
            if os.path.isdir(p):
                sample_names.append(filename)

        rgb_paths = []
        for sample_name in sample_names:
            rgb_paths.append(_rgb_path(root_dir, sample_name))

        super(Stage2TestSegDataset, self).__init__(range01, rgb_order, root_dir, sample_names, rgb_paths, None,
                                                   None, dummy=dummy, subset_indices=subset_indices)


