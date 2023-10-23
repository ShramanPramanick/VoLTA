import json
from tqdm import tqdm
from collections import defaultdict
from glob import glob


def path2rest(path, iid2captions):
    name = path.split("/")[-1]
    iid = int(name[:-4])

    cdicts = iid2captions[iid]
    captions = [c["phrase"] for c in cdicts]

    captions = list(set(captions))
    
    return captions

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
            if cap.strip(): 
                image_list.append(path)
                text_list.append(cap)

    return image_list, text_list

if __name__ == "__main__":
    i, t = get_id_list_vg("/datasets01/VisualGenome1.2/061517/")
    print(len(i))    
    print(len(t))