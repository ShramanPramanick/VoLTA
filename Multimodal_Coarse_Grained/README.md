## 🛠️ Environment Preparation

Clone the pretraining environment and install new dependencies

```bash
conda create -n volta_coarse_grained --clone volta
conda activate volta_coarse_grained
pip install -r requirements.txt
pip install -e .
```
Install sacred from source:
```
git clone https://github.com/IDSIA/sacred.git
cd sacred
python setup.py install
pip install numpy,pymongo
```

## ⚙️ Model Preparation

Convert volta pretrained checkpoint to make it compatible with fiber codebase using the following:
```
python convert_volta2fiber.py --old_model_path <volta_ckpt_path> --new_model_path <converted_ckpt_path>
```

## 📝 Data Preparation

We follow [ViLT](https://github.com/dandelin/ViLT) and [METER](https://github.com/zdou0830/METER) to prepare the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.

We have prepared the required data for quick start. We provide these as follows:
```
wget http://www.cis.jhu.edu/~shraman/VoLTA/datasets_pyarrow/NLVR2_arrows.zip
wget http://www.cis.jhu.edu/~shraman/VoLTA/datasets_pyarrow/VQAv2_arrows.zip
wget http://www.cis.jhu.edu/~shraman/VoLTA/datasets_pyarrow/mscoco2014_arrows.zip
wget http://www.cis.jhu.edu/~shraman/VoLTA/datasets_pyarrow/flickr30k_arrows.zip
```
Also, download the following file required for CIDEr optimization during captioning, and modify the path in the last line of config.
```
wget http://www.cis.jhu.edu/~shraman/VoLTA/coco-train-words.p
```


## 🎯 Fine-tuning on Downstream Tasks

### VQAv2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK

# training example
python run.py with data_root=<VQAv2_arrow_root> num_gpus=8 num_nodes=1 task_finetune_vqa per_gpu_batchsize=4 load_path=<pretrained_model_path>

# evaluation example
python run.py with data_root=<VQAv2_arrow_root> num_gpus=8 num_nodes=1 task_finetune_vqa per_gpu_batchsize=32 load_path=<finetuned_model_path> test_only=True
```

<strong>Note:</strong> The generated json file will look like 'vqa_submit_epoch=<epoch_number>-step=<setp_number>.json'. Submit this json in the challenge page to [eval.ai](https://eval.ai/web/challenges/challenge-page/830/overview) evaluation server to get the test-dev and/or test-std scores.

### NLVR2

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK

# training example
python run.py with data_root=<NLVR2_arrow_root> num_gpus=8 num_nodes=1 task_finetune_nlvr2 per_gpu_batchsize=4 load_path=<pretrained_model_path>

# evaluation example
python run.py with data_root=<NLVR2_arrow_root> num_gpus=8 num_nodes=1 task_finetune_nlvr2 per_gpu_batchsize=32 load_path=<finetuned_model_path> test_only=True
```  

### Flickr30k IR/TR

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK

# itc training example
python run.py with data_root=<flickr30k_arrow_root> num_gpus=8 num_nodes=1 task_finetune_irtr_itc_f30k per_gpu_batchsize=4 load_path=<pretrained_model_path>

# itc evaluation example
python run.py with data_root=<flickr30k_arrow_root> num_gpus=8 num_nodes=1 task_finetune_irtr_itc_f30k per_gpu_batchsize=32 load_path=<finetuned_model_path> get_recall_metric=True test_only=True
```

### COCO Captioning

```bash
export MASTER_ADDR=$DIST_0_IP
export MASTER_PORT=$DIST_0_PORT
export NODE_RANK=$DIST_RANK

# mle training example
python run.py with data_root=<coco_arrow_root> num_gpus=8 num_nodes=1 task_finetune_caption_mle_coco per_gpu_batchsize=4 load_path=<pretrained_model_path>

# mle+gold training example
python run.py with data_root=<coco_arrow_root> num_gpus=8 num_nodes=1 task_finetune_caption_gold_coco per_gpu_batchsize=4 load_path=<mle_finetuned_model_path>

# mle+cider optimization training example
python run.py with data_root=<coco_arrow_root> num_gpus=8 num_nodes=1 task_finetune_caption_cider_coco per_gpu_batchsize=1 load_path=<mle_finetuned_model_path>

# mle+gold+cider optimization training example
python run.py with data_root=<coco_arrow_root> num_gpus=8 num_nodes=1 task_finetune_caption_cider_coco per_gpu_batchsize=1 load_path=<gold_finetuned_model_path>

# mle evaluation example
python run.py with data_root=<coco_arrow_root> num_gpus=8 num_nodes=1 task_finetune_caption_mle_coco per_gpu_batchsize=32 load_path=<mle_finetuned_model_path> test_only=True

# gold evaluation example
python run.py with data_root=<coco_arrow_root> num_gpus=8 num_nodes=1 task_finetune_caption_gold_coco per_gpu_batchsize=32 load_path=<gold_finetuned_model_path> test_only=True

# cider evaluation example
python run.py with data_root=<coco_arrow_root> num_gpus=8 num_nodes=1 task_finetune_caption_cider_coco per_gpu_batchsize=32 load_path=<cider_finetuned_model_path> test_only=True

```

<strong>Note:</strong>
* Each evaluation will generate results which can be found in `results/caption.json`. Perform the 4 caption evaluation experiments serially.
* To compute scores on the generated captions, please clone [coco-caption](https://github.com/tylin/coco-caption/) repository, and run `coco-caption/cocoEvalCapDemo.py` after updating the `resFile` variable to the path containing `results/caption.json`.


## 🙏 Acknowledgements

Coarse-grained evaltuation pipeline is based on [FIBER](https://github.com/microsoft/FIBER).
