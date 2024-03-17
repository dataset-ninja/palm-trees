import csv
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    dataset_path = "/home/alex/DATASETS/TODO/palm trees/Palm-Counting-349images"
    batch_size = 30
    anns_ext = ".xml"
    ann_ext = "_labels.csv"

    def create_ann(image_path):
        labels = []

        ann_path = image_path.split(".")[0] + anns_ext

        tree = ET.parse(ann_path)
        root = tree.getroot()

        img_height = int(root.find(".//height").text)
        img_wight = int(root.find(".//width").text)

        if img_height == 3648 or img_height == 2160:
            drone = sly.Tag(drone_a)
        else:
            drone = sly.Tag(drone_b)

        file_name = get_file_name_with_ext(image_path)

        bboxes_data = name_to_boxes[file_name]
        for coords in bboxes_data:
            left = int(coords[0])
            top = int(coords[1])
            right = int(coords[2])
            bottom = int(coords[3])
            rect = sly.Rectangle(left=left, top=top, right=right, bottom=bottom)
            label = sly.Label(rect, obj_class)
            labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[drone])

    obj_class = sly.ObjClass("palm tree", sly.Rectangle, color=(255, 0, 0))
    drone_a = sly.TagMeta("DJI Phantom 4 Pro drone", sly.TagValueType.NONE)
    drone_b = sly.TagMeta("DJI Mavic Pro drone", sly.TagValueType.NONE)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(obj_classes=[obj_class], tag_metas=[drone_a, drone_b])
    api.project.update_meta(project.id, meta.to_json())

    for ds_name in os.listdir(dataset_path):

        data_path = os.path.join(dataset_path, ds_name)

        if dir_exists(data_path):

            dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

            name_to_boxes = defaultdict(list)
            ann_path = data_path + ann_ext
            with open(ann_path, "r") as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    name_to_boxes[row[0]].append(row[4:])

            images_names = [
                im_name for im_name in os.listdir(data_path) if get_file_ext(im_name) != anns_ext
            ]

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

            for images_names_batch in sly.batched(images_names, batch_size=batch_size):
                img_pathes_batch = [
                    os.path.join(data_path, image_name) for image_name in images_names_batch
                ]

                img_infos = api.image.upload_paths(dataset.id, images_names_batch, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                anns = [create_ann(image_path) for image_path in img_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(images_names_batch))

    return project
