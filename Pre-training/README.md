## 📝 Data Preparation

### Download Pre-training Dataset

- **COCO2014**: Download [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip) and [karpathy split](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).
```
data_root
    ├── train2014
    │   ├── COCO_train2014_000000250351.jpg
    │   ├── COCO_train2014_000000250352.jpg
    │   └── ...
    ├── val2014
    │   ├── COCO_val2014_000000165984.jpg
    │   ├── COCO_val2014_000000166003.jpg
    │   └── ...
    └── dataset_coco.json
```
<!--
- **COCO2017**: Our pre-training dataset, [mscoco2017](https://academictorrents.com/details/74dec1dd21ae4994dfd9069f9cb0443eb960c962) train split, is a dataset of ~120K image-caption pair. We download the dataset using [img2dataset](https://github.com/ShramanPramanick/img2dataset).
```
pip install img2dataset
wget https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/resolve/main/mscoco.parquet
img2dataset --url_list mscoco.parquet --input_format "parquet"\
            --url_col "URL" --caption_col "TEXT"\
            --output_folder mscoco --processes_count 16 --thread_count 64 --image_size 384\
            --enable_wandb True
``` 
- **SBU**: Follow these steps:
```
wget http://www.cis.jhu.edu/~shraman/sbu_url_captions.csv
img2dataset --url_list sbu_url_captions.csv --input_format "csv"\
            --url_col "url_col" --caption_col "caption_col"\ 
            --output_folder sbu --processes_count 16 --thread_count 64 --image_size 384\
```
-->
- **VG**: Download [image part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [image part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) and [region descriptions](http://visualgenome.org/static/data/dataset/region_descriptions.json.zip).
```
data_root
    ├── VG_100K
    │   ├── 10.jpg
    │   ├── 107899.jpg
    │   └── ...
    ├── VG_100K_2
    │   ├── 1.jpg
    │   ├── 100.jpg
    │   └── ...
    └── region_descriptions.json
```
## ⚓ Pre-trained Checkpoint
The pretrained weights of VoLTA can be downloaded as:
```bash
wget http://www.cis.jhu.edu/~shraman/VoLTA/ckpts/VoLTA_Pre-trained.pth
```


## 🏋️‍️ Pre-training
We use PyTorch’s native DistributedDataParallel (DDP) and FP16 mixed precision training. We pre-train our model for 200 epochs with a batch size of 256, using LARS optimizer.

- Pre-training with COCO + VG dataset
```
python main.py --batch_size 256 --epochs 200 --data_root_coco <coco_path> --data_root_vg <vg_path> --maxlen 30 --checkpoint-dir ./checkpoint/ --print_freq 100 --vg --name volta_pretraining
```

## 🙏Acknowledgement

This codebase is built on the [FIBER](https://github.com/microsoft/FIBER), [GOT](https://github.com/LiqunChen0606/Graph-Optimal-Transport) and [Barlow Twins](https://github.com/facebookresearch/barlowtwins) repository.

