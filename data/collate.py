# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random

#cfg=self.cfg,mask_generator=self.mask_generator,samples_list=data, n_tokens=self.n_tokens

def collate_data_and_cast(cfg,samples_list, n_tokens=None, mask_generator=None):
    #dtype = torch.half  
    mask_probability= cfg.ibot.mask_sample_probability
    mask_ratio_tuple=cfg.ibot.mask_ratio_min_max 
    #print('len(samples_list["local_crops"])): ', len(samples_list["local_crops"]))
    #print('len(samples_list["global_crops"])): ', len(samples_list["global_crops"]))
    #rint('samples_list["local_crops"][0].shape:  ', samples_list["local_crops"][0].shape)
    collated_global_crops = torch.cat(samples_list["global_crops"], dim=0) 

    collated_local_crops = torch.cat(samples_list["local_crops"], dim=0) 
    collated_gram_teacher_crops=None
    if "gram_teacher_crops" in samples_list: 
        collated_gram_teacher_crops = torch.cat(samples_list["gram_teacher_crops"], dim=0) 

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out= { 
        "collated_global_crops": collated_global_crops ,
        "collated_local_crops": collated_local_crops ,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops 
    return out 
 
def lejepa_collate_fn(batch):
    """ 
    input (batch):  
      batch['global_crops'] -> [Tensor(B, C, H, W), Tensor(B, C, H, W)]  
      
    output:
      Dictionary.
      'global_crops' -> Tensor(B, 2, C, H, W)
    """  
    return {
        
        'global_crops': torch.stack(batch['global_crops'], dim=1),  
        'local_crops': torch.stack(batch['local_crops'], dim=1)    
    }
