import os
import glob
import json
import random
import cv2
import numpy as np
from affine import Affine

default_annotation = {
    "version": "4.5.5",
    "flags": {},
    "shapes": [],
}

default_shape = {
    "group_id": None,
    "shape_type": "polygon",
    "flags": {}
}


def rand_affine(affine_range):
    """generate random affine transformation matrix

    Args:
        affine_range (dict): range of random affine in angle, scale, trans_x, trans_y, shear_x, shear_y

    Returns:
        ndarray: 3*3 affine transformation matrix
    """
    angle = random.uniform(affine_range['angle'][0], affine_range['angle'][1])
    scale = random.uniform(affine_range['scale'][0], affine_range['scale'][1])
    trans_x = random.uniform(affine_range['trans_x'][0], affine_range['trans_x'][1])
    trans_y = random.uniform(affine_range['trans_y'][0], affine_range['trans_y'][1])
    shear_x = random.uniform(affine_range['shear_x'][0], affine_range['shear_x'][1])
    shear_y = random.uniform(affine_range['shear_y'][0], affine_range['shear_y'][1])

    H = Affine.translation(trans_x, trans_y) *\
        Affine.shear(shear_x, shear_y) * \
        Affine.scale(scale) * \
        Affine.rotation(angle)
    return np.array(H).reshape((3, 3))


class RandomDefect:
    """crop annotated defect and compose new image containing defects randomly
    """

    def __init__(self, origin_dir, crop_dir, background_dir, compose_dir, image_ext, affine_range):
        """init

        Args:
            origin_dir (str): dir of original annotated images
            crop_dir (str): dir of cropped defect images
            background_dir (str): dir of background images, can be same as origin images
            compose_dir (str): dir of composed images
            image_ext (str): image ext name
            affine_range (dict): range of random affine
        """
        self.origin_dir = origin_dir
        self.crop_dir = crop_dir
        self.background_dir = background_dir
        self.compose_dir = compose_dir
        self.image_ext = image_ext
        self.affine_range = affine_range

        # all labels of defects
        self.labels = []

        # create output dirs
        for dir in [crop_dir, compose_dir]:
            if not os.path.exists(dir):
                os.mkdir(dir)

    def crop(self):
        """crop defects from original images
        """
        for json_fn in glob.glob(os.path.join(self.origin_dir, '*.json')):
            # load each annotation file
            with open(json_fn) as f:
                annotation = json.load(f)
            # load image file
            img_fn = json_fn.replace('.json', self.image_ext)
            img = cv2.imread(img_fn)

            # for each defect
            for i, shape in enumerate(annotation['shapes']):
                # add label to list if not in
                label = shape['label']
                if label not in self.labels:
                    self.labels.append(label)
                # create label dir if not exists
                crop_sub_dir = os.path.join(self.crop_dir, label)
                if not os.path.exists(crop_sub_dir):
                    os.mkdir(crop_sub_dir)

                # transform defect to rectangle
                src = np.array(shape['points']).astype(np.float32)
                w = int(np.linalg.norm(src[0]-src[1]))
                h = int(np.linalg.norm(src[2]-src[1]))
                dst = np.array([
                    [0, 0],
                    [w, 0],
                    [w, h],
                    [0, h],
                ], dtype=np.float32)
                H = cv2.getPerspectiveTransform(src, dst)
                crop = cv2.warpPerspective(img, H, (w, h))
                # save defect
                crop_fn = '%s_%d%s' % (os.path.splitext(os.path.basename(json_fn))[0], i, self.image_ext)
                cv2.imwrite(os.path.join(crop_sub_dir, crop_fn), crop)

    def compose(self, n_images, n_defects):
        """compose new images with defects

        Args:
            n_images (int): new image number
            n_defects (int): defect number add to each image
        """
        # all background filenames
        bg_fn_all = glob.glob(os.path.join(self.background_dir, '*'+self.image_ext))
        # all foreground filenames (cropped defects)
        fg_fn_all = {}
        for label in self.labels:
            fg_fn_all[label] = glob.glob(os.path.join(self.crop_dir, label, '*'+self.image_ext))

        for i in range(n_images):
            # choose random background image
            bg_fn = random.choice(bg_fn_all)
            bg = cv2.imread(bg_fn)

            # mask of all defects
            mask = np.zeros(bg.shape[:2], np.bool)

            # if annotation file exists
            json_fn = bg_fn.replace(self.image_ext, '.json')
            if os.path.exists(json_fn):
                with open(json_fn) as f:
                    annotation = json.load(f)
                # add defect bounding box to mask
                for shape in annotation['shapes']:
                    points = np.array(shape['points']).astype(np.int)
                    left, right = np.min(points[:, 0]), np.max(points[:, 0])
                    top, bottom = np.min(points[:, 1]), np.max(points[:, 1])
                    mask[top:bottom+1, left:right+1] = True
            # else create default annotation
            else:
                annotation = {
                    "imageHeight": bg.shape[0],
                    "imageWidth": bg.shape[1]
                }
                annotation.update(default_annotation)

            for _ in range(n_defects):
                while True:
                    # choose random label and foreground image
                    label = random.choice(self.labels)
                    fg_fn = random.choice(fg_fn_all[label])
                    fg = cv2.imread(fg_fn)

                    # apply random affine transformation
                    H = rand_affine(self.affine_range)
                    h, w = fg.shape[:2]
                    src = np.array([[
                        [0, 0],
                        [0, h],
                        [w, h],
                        [w, 0]
                    ]]).astype('float32')
                    dst = cv2.perspectiveTransform(src, H)
                    fg = cv2.warpPerspective(fg, H, (bg.shape[1], bg.shape[0]))

                    # if transformed defect overlaps current defects, choose another
                    m = fg != 0
                    m = m[:, :, 0] | m[:, :, 1] | m[:, :, 2]
                    if np.any(mask & m):
                        continue

                    # else update mask
                    mask |= m

                    # paste defect on background
                    m = np.stack([m, m, m], axis=2)
                    bg[m] = fg[m]

                    # add annotation
                    shape = {
                        'label': label,
                        'points': dst[0].tolist()
                    }
                    shape.update(default_shape)
                    annotation['shapes'].append(shape)
                    break
            # add annotations
            compose_fn = '%d%s' % (i, self.image_ext)
            annotation['imagePath'] = compose_fn
            annotation['imageData'] = None
            # save image file
            compose_fn = os.path.join(self.compose_dir, compose_fn)
            cv2.imwrite(compose_fn, bg)
            # save annotation file
            compose_json_fn = compose_fn.replace(self.image_ext, '.json')
            with open(compose_json_fn, 'w') as f:
                json.dump(annotation, f, indent=4)


if __name__ == '__main__':
    affine_range = {
        'angle': (-180, 180),
        'scale': (0.8, 1.2),
        'trans_x': (100, 900),
        'trans_y': (100, 900),
        'shear_x': (-10, 10),
        'shear_y': (-10, 10),
    }
    rp = RandomDefect('./origin', './crop', './background', './compose', '.png', affine_range)
    rp.crop()
    rp.compose(100, 5)
