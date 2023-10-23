import os
import pdb
import json
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from eda_nlp.code.eda import get_only_chars, eda

def get_id_list_separate(root_dir):

    image_list = []
    text_list = []
    for filename in glob(root_dir + '**/*.jpg', recursive=True):
        #image_list.append(filename)
        text_path = filename.split('.')[0] + '.txt'
        with open(text_path, "r") as f:
            caption = f.read()

        if len(caption.strip().split()) >= 2:
            text_list.append(caption)
            image_list.append(filename)

    return image_list, text_list

def get_id_list_vg(root_dir):

    with open(f"{root_dir}region_descriptions.json", "r") as fp:
        captions = json.load(fp)

    iid2captions = defaultdict(list)
    for cap in tqdm(captions):
        cap = cap["regions"]
        for c in cap:
            iid2captions[c["image_id"]].append(c)

    paths = list(glob(f"{root_dir}VG_100K/*.jpg")) + list(glob(f"{root_dir}VG_100K_2/*.jpg"))
    caption_paths = [path for path in paths if int(path.split("/")[-1][:-4]) in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(len(paths), len(caption_paths), len(iid2captions),)

    image_list, text_list = [], []

    for path in tqdm(caption_paths):
        captions = path2rest(path, iid2captions)
        for cap in captions:
            cap = get_only_chars(cap)
            if len(cap.strip().split()) >= 2: 
                image_list.append(path)
                text_list.append(cap)

    return image_list, text_list


def get_id_list_coco2014(root_dir):

    with open(f"{root_dir}dataset_coco.json", "r") as fp:
        captions = json.load(fp)

    captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()

    for cap in tqdm(captions):
        filename = cap["filename"]
        iid2split[filename] = cap["split"]
        for c in cap["sentences"]:
            iid2captions[filename].append(c["raw"])

    paths = list(glob(f"{root_dir}train2014/*.jpg")) + list(glob(f"{root_dir}val2014/*.jpg"))
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(len(paths), len(caption_paths), len(iid2captions),)

    image_list, text_list = [], []

    for path in tqdm(caption_paths):
        captions, split = path2rest_coco2014(path, iid2captions, iid2split)
        if ('train' in split) or ('restval' in split):
            for cap in captions:
                image_list.append(path)
                text_list.append(cap)
                  
    return image_list, text_list


def path2rest(path, iid2captions):
    name = path.split("/")[-1]
    iid = int(name[:-4])

    cdicts = iid2captions[iid]
    captions = [c["phrase"] for c in cdicts]

    captions = list(set(captions))
    
    return captions

def path2rest_coco2014(path, iid2captions, iid2split):
    name = path.split("/")[-1]
    captions = iid2captions[name]
    captions = list(set(captions))

    split = iid2split[name]

    return captions, split



def get_imagenet_list(root_dir):

    image_list = []
    for filename in glob.iglob(root_dir + '**/*.JPEG', recursive=True):
        image_list.append(filename)

    return image_list

def get_inat_list(root_json):

    image_path_list = []
    label_list = []
    f = open(root_json)
    data = json.load(f)

    for idx in range(len(data["images"])):
        image_path = data["images"][idx]["file_name"]
        label = int(image_path.split("/")[-2]) 
        image_path_list.append(image_path)
        label_list.append(label)

    return image_path_list, label_list

#image, text = get_id_list_separate(root_dir = '/export/r13/data/shraman/multimodal_self_supervision/datasets/cc3m/cc3m/')
#print(image[100], text[100])

if __name__=="__main__":

    #image_list_coco2014, text_list_coco2014 = get_id_list_coco2014(root_dir = "/export/io76a/data/shraman/multimodal_self_supervision/datasets/mscoco2014/")

#    image_list_sbu, text_list_sbu = get_id_list_separate(root_dir = '/export/io76a/data/shraman/multimodal_self_supervision/datasets/sbu/sbu/')
    image_list_vg, text_list_vg = get_id_list_vg(root_dir = '/export/io76a/data/shraman/multimodal_self_supervision/datasets/vg/')
    pdb.set_trace()
    print("=========================")


