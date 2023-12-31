import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pdb
import inspect

from . import swin_transformer, roberta
from . import heads, objectives, fiber_utils
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig
from .roberta import RobertaModel, _prepare_decoder_attention_mask


class FIBERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        bert_config = RobertaConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        resolution_after = config["image_size"]
        self.num_fuse_block = config["num_fuse_block"]
        self.num_text_layer = config["num_layers"]
        roberta.NUM_FUSE_BLOCK = swin_transformer.NUM_FUSE_BLOCK = self.num_fuse_block
        roberta.DIM_IMG = config["input_image_embed_size"]
        swin_transformer.DIM_TXT = config["input_text_embed_size"]

        self.cross_modal_text_transform = nn.Linear(config["input_text_embed_size"], config["hidden_size"])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config["input_image_embed_size"], config["hidden_size"])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.cross_modal_text_transform_itc = nn.Linear(config["input_text_embed_size"], config["hidden_size"])
        self.cross_modal_text_transform_itc.apply(objectives.init_weights)
        self.cross_modal_image_transform_itc = nn.Linear(config["input_image_embed_size"], config["hidden_size"])
        self.cross_modal_image_transform_itc.apply(objectives.init_weights)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                getattr(swin_transformer, self.hparams.config["vit"])(
                    pretrained=config["pretrained_vit"],
                    config=self.hparams.config,
                )
                RobertaModel.from_pretrained(config["tokenizer"])

            torch.distributed.barrier()

        self.vit_model = getattr(swin_transformer, self.hparams.config["vit"])(
            pretrained=config["pretrained_vit"],
            config=self.hparams.config,
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.text_transformer = RobertaModel.from_pretrained(config["tokenizer"])

        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(objectives.init_weights)
        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_text_pooler.apply(objectives.init_weights)
        self.itc_pooler = config["itc_pooler"]
        if self.itc_pooler:
            self.cross_modal_image_pooler_itc = heads.Pooler(config["hidden_size"])
            self.cross_modal_image_pooler_itc.apply(objectives.init_weights)
            self.cross_modal_text_pooler_itc = heads.Pooler(config["hidden_size"])
            self.cross_modal_text_pooler_itc.apply(objectives.init_weights)

        if (
            config["loss_names"]["mlm"] > 0
            or config["loss_names"]["caption_mle"] > 0
            or config["loss_names"]["caption_gold"] > 0
            or config["loss_names"]["caption_cider"] > 0
        ):
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"] * 2)
            self.itm_score.apply(objectives.init_weights)

        if (
            config["loss_names"]["caption_mle"] > 0
            or config["loss_names"]["caption_gold"] > 0
            or config["loss_names"]["caption_cider"] > 0
        ):
            self.cross_modal_att_layers = []
            for _ in range(self.num_text_layer - 2):  # the bottom layers will not be used
                linear_transform = nn.Linear(
                    config["input_image_embed_size"], int(config["input_image_embed_size"] / 2)
                )
                linear_transform.apply(objectives.init_weights)
                self.cross_modal_att_layers.append(linear_transform)
            self.cross_modal_att_layers = nn.ModuleList(self.cross_modal_att_layers)

            if config["loss_names"]["caption_cider"] > 0:
                from .cider.ciderD.ciderD import CiderD

                self.cider_scorer = CiderD(df=config["cider_path"])

        hs = self.hparams.config["hidden_size"]

        # ===================== Downstream ===================== #
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            state_dict = swin_adapt_position_encoding(
                state_dict, before=config["resolution_before"], after=resolution_after
            )
            self.load_state_dict(state_dict, strict=False)

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 4, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)

        '''
        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs*2, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False
        '''

        fiber_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
        text_only=False,
        image_only=False,
    ):
        if not text_only:
            if img is None:
                if f"image_{image_token_type_idx - 1}" in batch:
                    imgkey = f"image_{image_token_type_idx - 1}"
                else:
                    imgkey = "image"
                img = batch[imgkey][0]

        if not image_only:
            do_mlm = "_mlm" if mask_text else ""
            text_ids = batch[f"text_ids{do_mlm}"]
            text_labels = batch[f"text_labels{do_mlm}"]
            text_masks = batch[f"text_masks"]

        # block attn
        if text_only:
            text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
            device = text_embeds.device
            input_shape = text_masks.size()
            extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
            for layer_i, layer in enumerate(self.text_transformer.encoder.layer):
                text_embeds = layer(text_embeds, extend_text_masks)[0]

            text_embeds = self.cross_modal_text_transform_itc(text_embeds)

            if self.itc_pooler:
                cls_feats_text = self.cross_modal_text_pooler_itc(text_embeds)
            else:
                cls_feats_text = text_embeds[:, 0]

            cls_feats_text = cls_feats_text / cls_feats_text.norm(dim=-1, keepdim=True)

            ret = {
                "text_feats": text_embeds,
                "image_feats": None,
                "cls_feats": cls_feats_text,
                "text_labels": text_labels,
                "text_ids": text_ids,
                "text_masks": text_masks,
                "image": None,
            }

            return ret

        if image_only:
            image_embeds = self.vit_model.patch_embed(img)
            if self.vit_model.absolute_pos_embed is not None:
                image_embeds = image_embeds + self.vit_model.absolute_pos_embed
            image_embeds = self.vit_model.pos_drop(image_embeds)

            for layer_i, layer in enumerate(self.vit_model.layers):
                image_embeds = layer(image_embeds)
            image_embeds = self.vit_model.norm(image_embeds)
            image_embeds = self.cross_modal_image_transform_itc(image_embeds)
            image_feats = image_embeds

            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            if self.itc_pooler:
                cls_feats_image = self.cross_modal_image_pooler_itc(avg_image_feats)
            else:
                cls_feats_image = avg_image_feats[:, 0]

            cls_feats_image = cls_feats_image / cls_feats_image.norm(dim=-1, keepdim=True)

            ret = {
                "text_feats": None,
                "image_feats": image_embeds,
                "cls_feats": cls_feats_image,
                "text_labels": None,
                "text_ids": None,
                "text_masks": None,
                "image": None,
            }

            return ret

        image_embeds = self.vit_model.patch_embed(img)
        if self.vit_model.absolute_pos_embed is not None:
            image_embeds = image_embeds + self.vit_model.absolute_pos_embed
        image_embeds = self.vit_model.pos_drop(image_embeds)
        for layer_i, layer in enumerate(self.vit_model.layers[:2]):
            image_embeds = layer(image_embeds)

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_transformer.get_extended_attention_mask(text_masks, input_shape, device)
        num_pre_text = self.num_text_layer - self.num_fuse_block
        for layer_i, layer in enumerate(self.text_transformer.encoder.layer[:num_pre_text]):
            text_embeds = layer(text_embeds, extend_text_masks)[0]

        num_pre_block = 8 + num_pre_text
        for blk_cnt, blk in enumerate(self.vit_model.layers[2].blocks):
            if blk_cnt < num_pre_block:
                image_embeds = blk(image_embeds)
            else:
                fuse_image_embeds = blk(image_embeds, text_embeds, extend_text_masks)
                text_embeds = self.text_transformer.encoder.layer[blk_cnt - 8](
                    text_embeds, extend_text_masks, encoder_hidden_states=(image_embeds)
                )[0]
                image_embeds = fuse_image_embeds

        if self.vit_model.layers[2].downsample is not None:
            image_embeds = self.vit_model.layers[2].downsample(image_embeds)

        for blk_cnt, blk in enumerate(self.vit_model.layers[3].blocks):
            fuse_image_embeds = blk(image_embeds, text_embeds, extend_text_masks)
            text_embeds = self.text_transformer.encoder.layer[blk_cnt + 10](
                text_embeds, extend_text_masks, encoder_hidden_states=(image_embeds), last_norm=(blk_cnt == 0)
            )[0]
            image_embeds = fuse_image_embeds

        if self.vit_model.layers[3].downsample is not None:
            image_embeds = self.vit_model.layers[3].downsample(image_embeds)

        text_embeds = self.cross_modal_text_transform(text_embeds)
        image_embeds = self.cross_modal_image_transform(image_embeds)

        cls_feats_text = self.cross_modal_text_pooler(text_embeds)
        avg_image_feats = self.avgpool(image_embeds.transpose(1, 2)).view(image_embeds.size(0), 1, -1)
        cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_feats": text_embeds,
            "image_feats": image_embeds,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "image": img,
        }

        return ret

    def infer_caption(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        img=None,
        text_only=False,
        image_only=False,
        image_embeds=None,
    ):

        if image_embeds is None:
            if img is None:
                if f"image_{image_token_type_idx - 1}" in batch:
                    imgkey = f"image_{image_token_type_idx - 1}"
                else:
                    imgkey = "image"
                img = batch[imgkey][0]
            image_embeds = self.vit_model.patch_embed(img)
            if self.vit_model.absolute_pos_embed is not None:
                image_embeds = image_embeds + self.vit_model.absolute_pos_embed
            image_embeds = self.vit_model.pos_drop(image_embeds)
            for layer_i, layer in enumerate(self.vit_model.layers):
                image_embeds = layer(image_embeds)

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]

        text_embeds = self.text_transformer.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_ids.size()
        extend_text_masks = _prepare_decoder_attention_mask(text_masks, input_shape, text_embeds, device)
        for layer_i, layer in enumerate(self.text_transformer.encoder.layer):
            if layer_i < self.num_text_layer - self.num_fuse_block:
                text_embeds = layer(text_embeds, extend_text_masks)[0]
            elif layer_i < self.num_text_layer - 2:
                text_embeds = layer(
                    text_embeds,
                    extend_text_masks,
                    encoder_hidden_states=self.cross_modal_att_layers[layer_i](image_embeds),
                )[0]
            else:
                text_embeds = layer(text_embeds, extend_text_masks, encoder_hidden_states=image_embeds)[0]

        text_embeds = self.cross_modal_text_transform(text_embeds)
        cls_feats_text = self.cross_modal_text_pooler(text_embeds)
        cls_feats_image = cls_feats = cls_feats_text

        ret = {
            "text_feats": text_embeds,
            "image_embeds": image_embeds,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "image": img,
        }
        return ret

    '''
    def rank_breakpoint(self, rank=0):
        if torch.distributed.get_rank() != rank:
            torch.distributed.barrier()  # all processes wait here except the one we want to debug
        else:
            # the chosen rank will set a trace, but we want the frame from where this function is called
            pdb.Pdb().set_trace(inspect.currentframe().f_back.f_back)
            # once breakpoint gets skipped, synchronize with all other workers
            torch.distributed.barrier()
    '''

    def forward(self, batch):
        ret = dict()

        #if torch.distributed.get_rank() == 0:
            #print("Printing Batch Information....")
            #print(batch.keys())
            #print("Text", batch['text'])
            #print("Text_ids", batch['text_ids'])
            #print("Text_labels", batch['text_labels'])
            #print("Text_ids_mlm", batch['text_ids_mlm'])
            #print("Text_labels_mlm", batch['text_labels_mlm'])
            #print("Text_masks", batch['text_masks'])
            #print(len(batch['image']))
            #print(batch['image'])
            #print(batch['image'][0].shape)
        #self.rank_breakpoint(rank=0)
        #print("..............................")


        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Contrastive Loss
        if "itc" in self.current_tasks:
            ret.update(objectives.compute_itc(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            if "itc" in self.current_tasks:
                ret.update(objectives.compute_itm_hardneg(self, batch, ret["itc_i2t_logits"], ret["itc_t2i_logits"]))
            else:
                ret.update(objectives.compute_itm(self, batch))

        '''
        # IRTR
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))
        '''

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Captioning
        if "caption_mle" in self.current_tasks:
            ret.update(objectives.compute_caption_mle(self, batch))

        if "caption_gold" in self.current_tasks:
            ret.update(compute_caption_gold(self, batch))

        if "caption_cider" in self.current_tasks:
            ret.update(objectives.compute_caption_cider(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        fiber_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        fiber_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        fiber_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        fiber_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        fiber_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        if (
            self.hparams.config["loss_names"]["caption_mle"] > 0
            or self.hparams.config["loss_names"]["caption_gold"] > 0
            or self.hparams.config["loss_names"]["caption_cider"] > 0
        ):
            ret.update(objectives.caption_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)

        if (
            self.hparams.config["loss_names"]["caption_mle"] > 0
            or self.hparams.config["loss_names"]["caption_gold"] > 0
            or self.hparams.config["loss_names"]["caption_cider"] > 0
        ):
            objectives.caption_test_wrapup(outs, model_name)

        fiber_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return fiber_utils.set_schedule(self)


def compute_caption_gold(pl_module, batch, update_freq=1000, min_prob=0.1):
    tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    infer = pl_module.infer_caption(batch, mask_text=False, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])

    mlm_labels = infer["text_ids"]
    mlm_labels = torch.cat([mlm_labels[:, 1:], torch.ones_like(mlm_labels[:, :1]) * tokenizer.pad_token_id], 1)
    pad_mask = mlm_labels == tokenizer.pad_token_id

    if pl_module.training:
        if pl_module.global_step % update_freq == 0:
            if not hasattr(pl_module, "copy_step") or getattr(pl_module, "copy_step") < pl_module.global_step:
                setattr(pl_module, "copy_step", pl_module.global_step)
                if hasattr(pl_module, "copy_module"):
                    delattr(pl_module, "copy_module")
                setattr(pl_module, "copy_module", FIBERTransformerSS(pl_module.config))
                pl_module.copy_module.load_state_dict(pl_module.state_dict(), strict=False)
                pl_module.copy_module.to(pl_module.device)
                pl_module.copy_module.eval()

            torch.distributed.barrier()

        with torch.no_grad():
            pl_module.copy_module.eval()
            off_infer = pl_module.copy_module.infer_caption(batch, mask_text=False, mask_image=False)
            off_logits = pl_module.copy_module.mlm_score(off_infer["text_feats"])
            off_logits = torch.log(torch.nn.functional.softmax(off_logits, dim=-1) + 1e-9)

            bs, seq_len, vocab_size = off_logits.size()
            off_labels = mlm_labels.view(-1, 1)

            off_logits = off_logits.view(-1, vocab_size)
            off_probs = off_logits.gather(dim=-1, index=off_labels)
            off_probs = off_probs.view(bs, seq_len)
            off_probs = off_probs.exp()
            off_probs[pad_mask] = 0

            cur_sum = torch.zeros(bs, device=off_logits.device)
            cur_len = torch.zeros(bs, device=off_logits.device)
            denom = torch.ones(bs, device=off_logits.device)

            cum_prob = []
            for i in range(seq_len):
                cur_sum = cur_sum + off_probs[:, -i - 1]
                cur_len = cur_len + (mlm_labels[:, -i - 1] != tokenizer.pad_token_id).float()
                cur_prob = cur_sum / torch.max(cur_len, denom)
                cum_prob.append(cur_prob)

            cum_prob = torch.flip(torch.stack(cum_prob), [0]).transpose(0, 1)
            weights = cum_prob.detach() * off_probs.detach()
            weights = torch.max(weights, torch.ones_like(weights) * min_prob)
            weights = weights.contiguous().view(-1)

    mlm_labels[pad_mask] = -100

    mlm_loss = torch.nn.functional.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )

    if pl_module.training:
        weights = weights.view(bs, -1)
        mlm_loss = mlm_loss.view(bs, -1)
        mlm_loss = torch.sum(weights * mlm_loss, -1)
        mlm_loss = mlm_loss / (torch.sum(pad_mask, -1) + 1e-9)
        mlm_loss = torch.mean(mlm_loss)
    else:
        mlm_loss = torch.mean(mlm_loss)

    ret = {
        "caption_gold_loss": mlm_loss,
        "caption_gold_logits": mlm_logits,
        "caption_gold_labels": mlm_labels,
        "caption_gold_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_caption_gold_loss")(ret["caption_gold_loss"])
    acc = getattr(pl_module, f"{phase}_caption_gold_accuracy")(ret["caption_gold_logits"], ret["caption_gold_labels"])
    pl_module.log(f"caption_gold/{phase}/loss", loss)
    pl_module.log(f"caption_gold/{phase}/accuracy", acc)

    return ret
