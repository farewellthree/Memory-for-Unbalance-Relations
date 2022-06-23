import _init_paths
from datasets_rel import json_dataset_rel
from modeling_rel import model_builder_rel
from roi_data_rel.fast_rcnn_rel import get_prd_labels
from datasets_rel.roidb_rel import combined_roidb_for_training
from roi_data_rel.loader_rel import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
import utils_rel.net_rel as net_utils_rel
import utils.blob as blob_utils
import utils_rel.boxes_rel as box_utils_rel
from core.config import cfg
import numpy as np
import torch
import torch.nn.functional as F
import nn as mynn
from torch.autograd import Variable

def class_count(roidb):
    labels = []
    json_dataset_rel._add_prd_class_assignments(roidb)
    for entry in roidb:
        prd_label = get_prd_labels(entry)
        labels = np.append(labels,prd_label)
        print (prd_label)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num
    
    
def initialize_model(load_ckpt):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = model_builder_rel.Generalized_RCNN()
    model.train()
    model.cuda()

    load_name = load_ckpt
    checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
    net_utils_rel.load_ckpt_rel(model, checkpoint['model'])
    #model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)
    return model

def get_centroids():
    prase = 'vg'
    if prase == 'vg':
        cfg.MODEL.SUBTYPE = 3
        cfg.MODEL.USE_OVLP_FILTER = True
        cfg.MODEL.USE_FREQ_BIAS = True
        cfg.MODEL.NO_FC7_RELU = True
        cfg.MODEL.USE_SPATIAL_FEAT = True
        cfg.MODEL.ADD_SO_SCORES = True
        cfg.MODEL.ADD_SCORES_ALL = True
        cfg.MODEL.USE_BG = True
        cfg.MODEL.FASTER_RCNN = True
        cfg.MODEL.NUM_PRD_CLASSES = 50
        cfg.TRAIN.DATASETS = ('vg_train',)
        cfg.MODEL.CONV_BODY = 'VGG16.VGG16_conv_body'
        cfg.RPN.RPN_ON = True
        cfg.RPN.SIZES = (32, 64, 128, 256, 512)
        cfg.FAST_RCNN.ROI_BOX_HEAD = 'VGG16.VGG16_roi_conv5_head'
        cfg.MODEL.NUM_CLASSES = 151
        cfg.FAST_RCNN.ROI_XFORM_METHOD = 'RoIAlign'
        mix_centroids = torch.zeros(51,1024).cuda()
        #spt_centroids = torch.zeros(51,64).cuda()
        #spt_centroids = None
        #vis_centroids = torch.zeros(51,4096).cuda()
    elif prase == 'vrd':
        cfg.MODEL.SUBTYPE = 3
        cfg.MODEL.USE_OVLP_FILTER = True
        cfg.MODEL.USE_FREQ_BIAS = True
        cfg.MODEL.NO_FC7_RELU = True
        cfg.MODEL.USE_SPATIAL_FEAT = True
        cfg.MODEL.ADD_SO_SCORES = True
        cfg.MODEL.ADD_SCORES_ALL = True
        cfg.MODEL.USE_BG = True
        cfg.MODEL.FASTER_RCNN = True
        cfg.MODEL.CONV_BODY = 'VGG16.VGG16_conv_body'
        cfg.RPN.RPN_ON = True
        cfg.RPN.SIZES = (32, 64, 128, 256, 512)
        cfg.FAST_RCNN.ROI_BOX_HEAD = 'VGG16.VGG16_roi_conv5_head'
        cfg.MODEL.NUM_CLASSES = 101
        cfg.MODEL.NUM_PRD_CLASSES = 70
        cfg.TRAIN.DATASETS = ('vrd_train',)
        cfg.FAST_RCNN.ROI_XFORM_METHOD = 'RoIAlign'
        #spt_centroids = torch.zeros(71,64).cuda()
        #spt_centroids = None
        vis_centroids = torch.zeros(71,4096).cuda()
        #mix_centroids = torch.zeros(71,1024).cuda()
    path_str = '/home/wwt/ECCV/vg_combine/Outputs/e2e_faster_rcnn_VGG16_8_epochs_vg_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5_no_spt/origin/ckpt/model_step62722.pth'
    
    model = initialize_model(path_str)
    roidb, ratio_list, ratio_index = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, ())
    batchSampler = BatchSampler(
        sampler=MinibatchSampler(ratio_list, ratio_index),
        batch_size=1,
        drop_last=True
    )
    dataset = RoiDataLoader(
        roidb,
        101,
        training=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batchSampler,
        num_workers=1,
        collate_fn=collate_minibatch)
    dataiterator = iter(dataloader)
    #torch.save(model.RelDN.prd_cls_scores.state_dict(),'prd_score_vg.pkl')
    #torch.save(model.RelDN.spt_cls_scores.state_dict(),'spt_score_vg.pkl')
    torch.save(model.RelDN.mix_scores.state_dict(),'mix_score.pkl')
    #print (model.RelDN.mix_scores)
    with torch.set_grad_enabled(False):
        ii = 0
        labels = np.array([])
        while(True):
            try:
                input_data = next(dataiterator)
            except StopIteration:
                break
            print (ii)
            for key in input_data:
                if key != 'roidb':
                    input_data[key] = list(map(Variable, input_data[key]))
            data = input_data['data'][0]
            im_info = input_data['im_info'][0]
            roidb_ = input_data['roidb'][0]
            roidb_ = list(map(lambda x: blob_utils.deserialize(x)[0], roidb_))
            mix_cls_feat,prd_label = get_feat(model,data,im_info,roidb_)
            #vis_feat,spt_feat,prd_label = get_feat(model,data,im_info,roidb_)
            ii += 1
            labels = np.append(labels,prd_label)
            for i in range(len(prd_label)):
                label = prd_label[i]
                #spt_centroids[label] += spt_feat[i]
                #vis_centroids[label] += vis_feat[i]
                mix_centroids[label] += mix_cls_feat[i]
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    #spt_centroids /= torch.tensor(class_data_num).float().unsqueeze(1).cuda()
    #vis_centroids /= torch.tensor(class_data_num).float().unsqueeze(1).cuda()
    mix_centroids /= torch.tensor(class_data_num).float().unsqueeze(1).cuda()
    #return class_data_num,spt_centroids,vis_centroids
    return class_data_num,mix_centroids                   

def get_feat(model,data,im_info,roidb,dataset_name='vg_train'):
    im_data = data.cuda()
    device_id = im_data.get_device()
    blob_conv = model.Conv_Body(im_data)
    blob_conv_prd = model.Prd_RCNN.Conv_Body(im_data)
    rpn_ret = model.RPN(blob_conv, im_info, roidb)
    box_feat = model.Box_Head(blob_conv, rpn_ret, use_relu=True)
    cls_score, bbox_pred = model.Box_Outs(box_feat)
    use_relu = False
    
    fg_inds = np.where(rpn_ret['labels_int32'] > 0)[0]
    det_rois = rpn_ret['rois'][fg_inds]
    det_labels = rpn_ret['labels_int32'][fg_inds]
    det_scores = F.softmax(cls_score[fg_inds], dim=1)
    rel_ret = model.RelPN(det_rois, det_labels, det_scores, im_info, dataset_name, roidb)
    
    prd_label = rel_ret['all_prd_labels_int32']
    sbj_feat = model.S_Head(blob_conv, rel_ret, rois_name='sbj_rois', use_relu=use_relu)
    obj_feat = model.O_Head(blob_conv, rel_ret, rois_name='obj_rois', use_relu=use_relu)
    rel_feat = model.Prd_RCNN.Box_Head(blob_conv_prd, rel_ret, rois_name='rel_rois', use_relu=use_relu)
    spo_feat = torch.cat((sbj_feat, rel_feat, obj_feat), dim=1)
    spt_feat = rel_ret['spt_feat']
    if spo_feat.dim() == 4:
        spo_feat = spo_feat.squeeze(3).squeeze(2)
    prd_cls_feats = model.RelDN.prd_cls_feats(spo_feat)
    spt_feat = Variable(torch.from_numpy(spt_feat.astype('float32'))).cuda(device_id)
    spt_cls_feats = model.RelDN.spt_cls_feats(spt_feat)
    #spt_cls_feats = None
    mix_feat = torch.cat((prd_cls_feats,spt_cls_feats),dim=1)
    mix_cls_feat = model.RelDN.mix_feats(mix_feat)
    #return prd_cls_feats,spt_cls_feats,prd_label
    return mix_cls_feat,prd_label

if __name__=='__main__':
    #class_data_num,vis_centroids = get_centroids()
    class_data_num,mix_centroids = get_centroids()
    print (class_data_num,len(class_data_num))
    #np.save('spt_centroids_vrd_version1.npy',spt_centroids.cpu().numpy())
    #np.save('vis_centroids_vrd_version1.npy',vis_centroids.cpu().numpy())
    np.save('mix_centroids.npy',mix_centroids.cpu().numpy())



    
##    roidb, ratio_list, ratio_index = combined_roidb_for_training(
##        ('vg_train',), ())
##    print (len(roidb))
##    a = class_count(roidb)
##    print (a,len(a))
##    np.save('a.npy',a)
    
        
    
