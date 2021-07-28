#更新了192时出现的错误
import os
from time import time

import torch
import torch.nn.functional as F

import numpy as np
import SimpleITK as sitk
import xlsxwriter as xw
import scipy.ndimage as ndimage

from net.ResUnet_dice import Net

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

val_ct_dir = './data_ori/TJ/'
#val_seg_dir = './Training/GT/'

organ_pred_dir = './data_ori/TJ_spleen/'

module_dir = './module/model_best/FINAL_liver_net_2_1200-0.026-0.153.pth'

upper = 350
lower = -upper
down_scale = 0.5
size = 48
slice_thickness = 3


organ_list = [
    'liver',

]

# 定义网络并加载参数
net = torch.nn.DataParallel(Net(training=False)).cuda()
net.load_state_dict(torch.load(module_dir))
net.eval()


# 开始正式进行测试
for file_index, file in enumerate(os.listdir(val_ct_dir)):

    start_time = time()

    # 将CT读入内存
    ct = sitk.ReadImage(os.path.join(val_ct_dir, file), sitk.sitkInt16)
    ct_array_ori = sitk.GetArrayFromImage(ct)
    ct_array= sitk.GetArrayFromImage(ct)
    print(file)

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    # 对CT使用双三次算法进行插值，插值之后的array依然是int16
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)
    print(ct_array.shape)

    if ct_array.shape[0]%48==0:
        print(file)
        # 在轴向上进行切块取样
        flag = False
        start_slice = 0
        end_slice = start_slice + size - 1
        ct_array_list = []
    
        while end_slice <= ct_array.shape[0] - 1:
            ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])
    
            start_slice = end_slice + 1
            end_slice = start_slice + size - 1
    
        ## 当无法整除的时候反向取最后一个block
        #if end_slice is not ct_array.shape[0] - 1:
        #    flag = True
        #    count = ct_array.shape[0] - start_slice
        #    ct_array_list.append(ct_array[-size:, :, :])
    else:
        print(file,"245")
        # 在轴向上进行切块取样
        flag = False
        start_slice = 0
        end_slice = start_slice + size - 1
        ct_array_list = []
        while end_slice <= ct_array.shape[0] - 1:
            ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])
    
            start_slice = end_slice + 1
            end_slice = start_slice + size - 1
        # 当无法整除的时候反向取最后一个block
        if end_slice is not ct_array.shape[0] - 1:
            flag = True
            count = ct_array.shape[0] - start_slice
            ct_array_list.append(ct_array[-size:, :, :])
            
    outputs_list = []
    with torch.no_grad():
        for ct_array in ct_array_list:

            ct_tensor = torch.FloatTensor(ct_array).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0)
            ct_tensor = ct_tensor.unsqueeze(dim=0)

            outputs = net(ct_tensor)
            outputs = outputs.squeeze()

            # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
            outputs_list.append(outputs.cpu().detach().numpy())
            del outputs

    # 执行完之后开始拼接结果
    pred_seg = np.concatenate(outputs_list[0:-1], axis=1)
    if flag is False:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=1)
    else:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1][:, -count:, :, :]], axis=1)

#    # 将金标准读入内存来计算dice系数
#    seg = sitk.ReadImage(os.path.join(val_seg_dir, file.replace('img', 'label')), sitk.sitkUInt8)
#    seg_array = sitk.GetArrayFromImage(seg)

    # 使用线性插值将预测的分割结果缩放到原始nii大小
    pred_seg = torch.FloatTensor(pred_seg).unsqueeze(dim=0)
    pred_seg = F.upsample(pred_seg, ct_array_ori.shape, mode='trilinear').squeeze().detach().numpy()
    pred_seg = np.argmax(pred_seg, axis=0)
    pred_seg = np.round(pred_seg).astype(np.uint8)

    print('size of pred: ', pred_seg.shape)


    # 将预测的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(pred_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(organ_pred_dir, file.replace('img', 'liver')))
    del pred_seg

    speed = time() - start_time

#    worksheet.write(14, file_index + 1, speed)

    print('this case use {:.3f} s'.format(speed))
    print('-----------------------')


# 最后安全关闭表格
#workbook.close()

        

    

