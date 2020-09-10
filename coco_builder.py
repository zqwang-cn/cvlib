import json
import os
import shutil
import imagesize


class CocoBuilder:
    """coco dataset builder
    """
    def __init__(self, root, task='instances', year=2017, copy_link=True):
        """init

        Args:
            root (str): root dir to place coco dataset
            task (str, optional): coco dataset task. Defaults to 'instances'.
            year (int, optional): coco dataset year. Defaults to 2017.
            copy_link (bool, optional): only copy image link to coco dataset (set False to copy entire image). Defaults to True.
        """
        self.root = root
        self.task = task
        self.year = year
        self.copy_link = copy_link
        self.categories = []
        if os.path.exists(root):
            shutil.rmtree(root)
        os.mkdir(root)
        os.mkdir(os.path.join(root, 'annotations'))

    def start(self, split):
        """start building coco dataset of one split

        Args:
            split (str): split start to build (train, val, test)
        """
        self.split = split
        self.image_id = 0
        self.ann_id = 0
        self.images = []
        self.anns = []
        self.split_path = '%s/%s%d' % (self.root, split, self.year)
        if not os.path.exists(self.split_path):
            os.mkdir(self.split_path)

    def add_image(self, filename):
        """add one image to current split

        Args:
            filename (str): image filename
        """
        basename = os.path.basename(filename)
        width, height = imagesize.get(filename)
        self.image_id += 1
        self.images.append({
            'id': self.image_id,
            'file_name': basename,
            'width': width,
            'height': height
            })
        if self.copy_link:
            os.symlink(os.path.relpath(filename, self.split_path), os.path.join(self.split_path, basename))
        else:
            shutil.copy2(filename, self.split_path)

    def add_annotation(self, bbox, c):
        """add one annotation to current image

        Args:
            bbox (list): bounding box of annotation
            c (str): category of annotation
        """
        if c not in self.categories:
            self.categories.append(c)
        self.ann_id += 1
        self.anns.append({
            'id': self.ann_id,
            'image_id': self.image_id,
            'bbox': bbox,
            'category_id': self.categories.index(c)+1,
            })

    def save(self):
        """save current split
        """
        categories = []
        for i, c in enumerate(self.categories):
            categories.append({
                'id': i+1,
                'name': c
            })
        coco = {
            'images': self.images,
            'annotations': self.anns,
            'categories': categories
        }
        coco_file = '%s/annotations/%s_%s%d.json' % (self.root, self.task, self.split, self.year)
        with open(coco_file, 'w') as f:
            json.dump(coco, f, indent=4)
