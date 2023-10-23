import torch
import argparse
import json
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VoLTA to FIBER model conversion')
    parser.add_argument('--old_model_path', default='volta_ckpt.tar', type=str, help='VoLTA Model path')
    parser.add_argument('--new_model_path', default='converted_model.ckpt', type=str, help='converted model path')
    parser.add_argument('--fiber_keys', default='fiber_keys.json', type=str, help='fiber keys path')

    args = parser.parse_args()

    old_model = torch.load(args.old_model_path, map_location='cpu')
    old_list = list(old_model['model'].keys())
    fiber_model = copy.deepcopy(old_model)

    f = open(args.fiber_keys, 'r')
    fiber_model['state_dict'] = json.load(f)

    try:
        fiber_model.pop('epoch')
    except:
        pass
    try:
        fiber_model.pop('model')
    except:
        pass
    try:
        fiber_model.pop('optimizer')
    except:
        pass

    fiber_model['state_dict']['cross_modal_text_transform_itc.weight'] = torch.zeros(768, 768)
    fiber_model['state_dict']['cross_modal_text_transform_itc.bias'] = torch.zeros(768)
    fiber_model['state_dict']['cross_modal_image_transform_itc.bias'] = torch.zeros(768)
    fiber_model['state_dict']['cross_modal_text_pooler_itc.dense.weight'] = torch.zeros(768, 768)
    fiber_model['state_dict']['cross_modal_text_pooler_itc.dense.bias'] = torch.zeros(768)
    fiber_model['state_dict']['cross_modal_image_pooler_itc.dense.weight'] = torch.zeros(768, 768)
    fiber_model['state_dict']['cross_modal_image_transform_itc.weight'] = torch.zeros(768, 1024)
    fiber_model['state_dict']['token_type_embeddings.weight'] = torch.zeros(2, 768)
    fiber_model['state_dict']['cross_modal_image_pooler_itc.dense.bias'] = torch.zeros(768)

    for key in old_list:
        previous_key = key
        if key in list(fiber_model['state_dict'].keys()):
            fiber_model['state_dict'][key] = old_model['model'][previous_key]

    torch.save(fiber_model, args.new_model_path)
