import torch
import torch.nn as nn
import torch.nn.functional as F
import swin_transformer, roberta
from swin_helpers import swin_adapt_position_encoding
from roberta import RobertaModel, _prepare_decoder_attention_mask
import heads
from transformers import RobertaConfig
import numpy as np

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class Model(nn.Module):
    def __init__(self, config=None, feature_dim = 1024, projector_neurons = 2048, text_model_name = 'roberta-base', maxlen = 30, task_names = 'BTGOT, MLM, ITM'):
        super(Model, self).__init__()
        self.input_feat_map = {'image':1024,'text':768}
        self.projector_neurons = projector_neurons
        self.task_names = task_names

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                getattr(swin_transformer, "swin_base_patch4_window12_384_in22k")(
                    pretrained=True
                )
                RobertaModel.from_pretrained("roberta-base")

            torch.distributed.barrier()

        # image encoder
        self.vit_model = getattr(swin_transformer, "swin_base_patch4_window12_384_in22k")(pretrained=True)
        self.avgpool_img = nn.AdaptiveAvgPool2d((1, 1024)) # for BT GOT
        # image projection head for BT
        self.g_img = nn.Sequential(nn.Linear(self.input_feat_map['image'], self.projector_neurons, bias=False), nn.BatchNorm1d(self.projector_neurons),
                                               nn.ReLU(inplace=True), nn.Linear(self.projector_neurons, self.projector_neurons, bias=True), nn.BatchNorm1d(self.projector_neurons),
                                               nn.ReLU(inplace=True), nn.Linear(self.projector_neurons, feature_dim, bias=True))
        # image projection for GOT
        self.inter_g_1_img = nn.Linear(self.input_feat_map['image'], self.projector_neurons, bias=False)
        self.inter_g_2_img = nn.BatchNorm1d(self.projector_neurons)
        self.inter_g_3_img = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(self.projector_neurons, self.projector_neurons, bias=True)) ## The hard-coded dimensions need to be tuned
        self.inter_g_4_img = nn.BatchNorm1d(self.projector_neurons)
        self.inter_g_5_img = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(self.projector_neurons, feature_dim, bias=True)) ## The hard-coded dimensions need to be tuned


        # text encoder
        self.text_transformer = RobertaModel.from_pretrained(text_model_name) # "roberta-base"
        self.avgpool_text = nn.AdaptiveAvgPool2d((1, 768)) # for BT GOT

        # text projection head for BT
        self.g_text = nn.Sequential(nn.Linear(self.input_feat_map['text'], self.projector_neurons, bias=False), nn.BatchNorm1d(self.projector_neurons),
                                               nn.ReLU(inplace=True), nn.Linear(self.projector_neurons, self.projector_neurons, bias=True), nn.BatchNorm1d(self.projector_neurons),
                                               nn.ReLU(inplace=True), nn.Linear(self.projector_neurons, feature_dim, bias=True))
        # text projection for GOT
        self.inter_g_1_text = nn.Linear(self.input_feat_map['text'], self.projector_neurons, bias=False)
        self.inter_g_2_text = nn.BatchNorm1d(self.projector_neurons)
        self.inter_g_3_text = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(self.projector_neurons, self.projector_neurons, bias=True)) ## The hard-coded dimensions need to be tuned
        self.inter_g_4_text = nn.BatchNorm1d(self.projector_neurons)
        self.inter_g_5_text = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(self.projector_neurons, feature_dim, bias=True)) ## The hard-coded dimensions need to be tuned

        if ('MLM' in self.task_names or 'ITM' in self.task_names):
            # for FIBER-like cross-attention
            # need to make sure how to get the config file
            # self.save_hyperparameters() # to be checked
            self.config = config

            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=maxlen,
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

            self.num_fuse_block = config["num_fuse_block"]
            self.num_text_layer = config["num_layers"]
            roberta.NUM_FUSE_BLOCK = swin_transformer.NUM_FUSE_BLOCK = self.num_fuse_block
            roberta.DIM_IMG = config["input_image_embed_size"]
            swin_transformer.DIM_TXT = config["input_text_embed_size"]

            self.cross_modal_text_transform = nn.Linear(config["input_text_embed_size"], config["hidden_size"])
            self.cross_modal_text_transform.apply(init_weights)
            self.cross_modal_image_transform = nn.Linear(config["input_image_embed_size"], config["hidden_size"])
            self.cross_modal_image_transform.apply(init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


            self.avgpool = nn.AdaptiveAvgPool1d(1)

            self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"])
            self.cross_modal_image_pooler.apply(init_weights)
            self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
            self.cross_modal_text_pooler.apply(init_weights)

        if 'MLM' in self.task_names:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(init_weights)

        if 'ITM' in self.task_names:
            self.itm_score = heads.ITMHead(config["hidden_size"] * 2)
            self.itm_score.apply(init_weights)

    def forward(self, img_x, text_x, false_image_0=None, text_mlm_ids=None, text_masks=None, device='cuda'):

        ret = {}

        if 'BTGOT' in self.task_names:
            # text transformer
            if text_masks is None:
                raise NameError('attention masks cannot be None for text')
            text_x_2 = self.text_transformer(input_ids=text_x, attention_mask=text_masks, return_dict=True)
            text_local_feature = text_x_2.last_hidden_state
            text_feature = self.avgpool_text(text_local_feature).squeeze(dim=1)

            # text GOT
            text_local_feature = self.inter_g_3_text(self.inter_g_2_text(self.inter_g_1_text(text_local_feature).permute(0,2,1)).permute(0,2,1))
            text_local_feature = self.inter_g_5_text(self.inter_g_4_text(text_local_feature.permute(0,2,1)).permute(0,2,1))

            # text BT
            text_out = self.g_text(text_feature)

            # swin transformer
            img_local_feature = self.vit_model(img_x)
            img_feature = self.avgpool_img(img_local_feature).squeeze(dim=1)

            # swin GOT
            img_local_feature = self.inter_g_3_img(self.inter_g_2_img(self.inter_g_1_img(img_local_feature).permute(0,2,1)).permute(0,2,1))
            img_local_feature = self.inter_g_5_img(self.inter_g_4_img(img_local_feature.permute(0,2,1)).permute(0,2,1))

            # swin BT
            img_out = self.g_img(img_feature)

            ret.update({'image_feats':F.normalize(img_feature, dim=-1),
                'image_local_feats':F.normalize(img_local_feature, dim=-1),
                'image_out':F.normalize(img_out, dim=-1),
                'text_feats':F.normalize(text_feature, dim=-1),
                'text_local_feats':F.normalize(text_local_feature, dim=-1),
                'text_out':F.normalize(text_out, dim=-1)
                })

        if 'ITM' in self.task_names:
            # FIBER-like cross-attention
            pos_len = text_x.size(0) // 2
            neg_len = text_x.size(0) - pos_len

            itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(text_x.device)
            itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]
                
        
            assert len(itm_labels) >= 2, "ITM loss cannot be computed with per gpu batchsize = 1"

            # itm_images = []
            false_idx = []
            for idx in range(len(itm_labels)):
                if itm_labels[idx] == 1:
                    false_idx.append(idx)
                else:
                    if idx==0 and len(itm_labels)<=3:
                        if len(itm_labels)==2:
                            false_idx.append(1)
                        elif len(itm_labels)==3:
                            false_idx.append(np.random.randint(0,2))
                        #false_idx.append(np.random.randint(idx+1,len(itm_labels)-1))
                    elif idx==1 and len(itm_labels)<=3:
                        if len(itm_labels)==2:
                            false_idx.append(0)
                        elif len(itm_labels)==3:
                            false_idx.append(0) if np.random.rand() > 0.5 else false_idx.append(2)
                    elif (idx==0 or idx==1) and len(itm_labels)>=4:
                        false_idx.append(np.random.randint(idx+1,len(itm_labels)))
                    else:
                        false_idx.append(np.random.randint(idx+1,len(itm_labels)) if (np.random.rand() > 0.5 and idx<len(itm_labels)-2) else np.random.randint(0,idx))
                   # itm_images.append(img_x[false_idx])
            #itm_images = [torch.stack(itm_images, dim=0)]


            image_embeds = self.vit_model.patch_embed(img_x[false_idx]) # itm_images for itm instead of img_x

            if self.vit_model.absolute_pos_embed is not None:
                image_embeds = image_embeds + self.vit_model.absolute_pos_embed
            image_embeds = self.vit_model.pos_drop(image_embeds)
            for layer_i, layer in enumerate(self.vit_model.layers[:2]):
                image_embeds = layer(image_embeds)

            text_embeds = self.text_transformer.embeddings(input_ids=text_x) # before it was input_ids=text_ids
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

            ret.update({
                "cross_attn_itm_logits": self.itm_score(cls_feats),
                "cross_attn_itm_labels": itm_labels
            })

        if 'MLM' in self.task_names:
            # FIBER-like cross-attention
            image_embeds = self.vit_model.patch_embed(img_x)
            if self.vit_model.absolute_pos_embed is not None:
                image_embeds = image_embeds + self.vit_model.absolute_pos_embed
            image_embeds = self.vit_model.pos_drop(image_embeds)
            for layer_i, layer in enumerate(self.vit_model.layers[:2]):
                image_embeds = layer(image_embeds)

            text_embeds = self.text_transformer.embeddings(input_ids=text_mlm_ids) # before it was input_ids=text_ids
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

            ret.update({
                "cross_attn_mlm_logits": self.mlm_score(text_embeds)
            })

        return ret
