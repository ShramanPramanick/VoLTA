## 🛠️ Environment Preparation

Clone the pretraining environment and install new dependencies

```bash
conda create -n volta_fine_grained --clone volta
conda activate volta_fine_grained
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

Download the initialization weights of FPN and DyHead from [here](http://www.cis.jhu.edu/~shraman/VoLTA/ckpts/FPN_DyHead_init.pkl) and place it in <path_to_fpn_dyhead_init>. 
Convert the VoLTA pretrained checkpoint to make it compatible with FIBER codebase using the following:
```
python convert_volta2fiber.py --old_model_path <volta_ckpt_path> --new_model_path <converted_ckpt_path> --fpn_dyhead_init <path_to_fpn_dyhead_init>
```

## 📝 Data Preparation

We follow [FIBER](https://github.com/microsoft/FIBER) to prepare the datasets. See [this link](https://github.com/microsoft/FIBER/blob/main/fine_grained/DATA.md) for details. We also provide the config files in the `configs` folder.


## 🎯 Fine-tuning on Downstream Tasks

### REC

```bash
# training example for RefCOCO
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py \
--config-file configs/refcoco.yaml \
--skip-test \
MODEL.WEIGHT <pretrained_model_path> \
OUTPUT_DIR <output_dir_path>

# evaluation example for RefCOCO testB
python tools/test_grounding_net.py \
--config-file configs/refcoco.yaml \
--task_config configs/refexp/_refcoco_testB.yaml \
--weight <finetuned_model_path> \
OUTPUT_DIR <output_dir_path> \
TEST.IMS_PER_BATCH 1 \
SOLVER.IMS_PER_BATCH 1 \
TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM -1 \
TEST.EVAL_TASK grounding \
MODEL.ATSS.PRE_NMS_TOP_N 3000 \
MODEL.ATSS.DETECTIONS_PER_IMG 100 \
MODEL.ATSS.INFERENCE_TH 0.0
```

<strong>Note:</strong>
The above examples are shown for RefCOCO training and RefCOCO testB evaluation. For training and evaluation on other splits, please change the name of the config file accordingly.

### COCO Det

```bash
# training example
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py \
--config-file configs/e2e_dyhead_SwinT_B_FPN_coco_finetuning_fusion_backbone.yaml \
--skip-test \
MODEL.WEIGHT <pretrained_model_path> \
OUTPUT_DIR <output_dir_path>

# evaluation example
python tools/test_grounding_net.py \
--config-file configs/e2e_dyhead_SwinT_B_FPN_coco_finetuning_fusion_backbone.yaml \
--weight <finetuned_model_path> \
TEST.IMS_PER_BATCH 1 \
TEST.EVAL_TASK detection \
OUTPUT_DIR <output_dir_path>
```  

### LVIS Det

```bash
# training example
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py \
--config-file configs/e2e_dyhead_SwinT_B_FPN_lvis_finetuning_fusion_backbone.yaml \
--skip-test \
MODEL.WEIGHT <pretrained_model_path> \
OUTPUT_DIR <output_dir_path>

# evaluation example
python -m torch.distributed.launch --nproc_per_node=4 \
tools/test_grounding_net.py \
--config-file configs/e2e_dyhead_SwinT_B_FPN_lvis_finetuning_fusion_backbone.yaml \
--task_config configs/lvis/minival.yaml \
--weight <finetuned_model_path> \
TEST.EVAL_TASK detection OUTPUT_DIR <output_dir_path> \
TEST.CHUNKED_EVALUATION 40  TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 4 TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 3000 MODEL.RETINANET.DETECTIONS_PER_IMG 300 MODEL.FCOS.DETECTIONS_PER_IMG 300 MODEL.ATSS.DETECTIONS_PER_IMG 300 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 300
```

## 🙏 Acknowledgements

Fine-grained evaltuation pipeline is based on [FIBER](https://github.com/microsoft/FIBER).
