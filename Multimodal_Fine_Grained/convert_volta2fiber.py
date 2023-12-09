import torch
import argparse
import pickle
import json
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VoLTA to FIBER model conversion')
    parser.add_argument('--old_model_path', type=str, default='volta_ckpt.tar', help='VoLTA model path')
    parser.add_argument('--new_model_path', type=str, default='converted_model.ckpt', help='converted model path')
    parser.add_argument('--fiber_keys', type=str, default='fiber_keys.json', help='fiber keys path')
    parser.add_argument('--fpn_dyhead_init', type=str, default='FPN_DyHead_init.pkl', help='dyhead init path')

    args = parser.parse_args()

    old_model = torch.load(args.old_model_path)
    old_list = list(old_model['model'].keys())

    f = open(args.fiber_keys, 'r')
    fiber_keys = json.load(f)
    fiber_keys = fiber_keys['keys']
    fiber_dict = {}
    for k in fiber_keys:
        fiber_dict[k] = []
    
    f2 = open(args.fpn_dyhead_init, 'rb')
    fpn_dyhead_init_dict = pickle.load(f2)
    
    for key in old_list:
        previous_key = key
        if 'vit_model' in key:
            key = key.replace('vit_model', 'fusion_backbone.backbone.body')
        elif 'text_transformer' in key:
            key = key.replace('text_transformer', 'fusion_backbone.language_backbone.body.model')
        else:
            pass

        if key in fiber_keys:
            fiber_dict[key] = old_model['model'][previous_key]
        else:
            pass

    for key in list(fpn_dyhead_init_dict.keys()):
        fiber_dict[key] = fpn_dyhead_init_dict[key]

    fiber_model = {'state_dict': {}}
    for key in fiber_keys:
        new_key = 'module.' + str(key)
        fiber_model['state_dict'].update({new_key: fiber_dict[key]})
    
    torch.save(fiber_model, args.new_model_path)

