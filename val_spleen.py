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

val_ct_dir = './Path of original images/'
val_seg_dir = './Path of ground truth/'

organ_pred_dir = './Path to save segmentation/'

module_dir = './FINAL_spleen_net.pth'

upper = 350
lower = -upper
down_scale = 0.5
size = 48
slice_thickness = 3


organ_list = [
    'spleen',

]

# 创建一个表格对象，并添加一个sheet，后期配合window的excel来出图
workbook = xw.Workbook('./spleen_task_result.xlsx')
worksheet = workbook.add_worksheet('result')

# 设置单元格格式
bold = workbook.add_format()
bold.set_bold()

center = workbook.add_format()
center.set_align('center')

center_bold = workbook.add_format()
center_bold.set_bold()
center_bold.set_align('center')

worksheet.set_column(1, len(os.listdir(val_ct_dir)), width=15)
worksheet.set_column(0, 0, width=30, cell_format=center_bold)
worksheet.set_row(0, 20, center_bold)

# 写入文件名称
worksheet.write(0, 0, 'file name')
for index, file_name in enumerate(os.listdir(val_ct_dir), start=1):
    worksheet.write(0, index, file_name)

# 写入各项评价指标名称
for index, organ_name in enumerate(organ_list, start=1):
    worksheet.write(index, 0, organ_name)
worksheet.write(14, 0, 'speed')
worksheet.write(15, 0, 'shape')
worksheet.write(16, 0, 'ACC')
#worksheet.write(17, 0, 'PPV')
worksheet.write(18, 0, 'JAC')
#worksheet.write(19, 0, 'SPEC')

# 定义网络并加载参数
net = torch.nn.DataParallel(Net(training=False)).cuda()
net.load_state_dict(torch.load(module_dir))
net.eval()


# 开始正式进行测试
for file_index, file in enumerate(os.listdir(val_ct_dir)):

    start_time = time()

    # 将CT读入内存
    ct = sitk.ReadImage(os.path.join(val_ct_dir, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    print(file)

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    # 对CT使用双三次算法进行插值，插值之后的array依然是int16
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)

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

    # 将金标准读入内存来计算dice系数
    seg = sitk.ReadImage(os.path.join(val_seg_dir, file.replace('img', 'label')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # 使用线性插值将预测的分割结果缩放到原始nii大小
    pred_seg = torch.FloatTensor(pred_seg).unsqueeze(dim=0)
    pred_seg = F.upsample(pred_seg, seg_array.shape, mode='trilinear').squeeze().detach().numpy()
    pred_seg = np.argmax(pred_seg, axis=0)
    pred_seg = np.round(pred_seg).astype(np.uint8)

    print('size of pred: ', pred_seg.shape)
    print('size of GT: ', seg_array.shape)

    worksheet.write(15, file_index + 1, pred_seg.shape[0])

    # 计算每一种器官的dice系数，并将结果写入表格中存储
    for organ_index, organ in enumerate(organ_list, start=1):

        pred_organ = np.zeros(pred_seg.shape)
        target_organ = np.zeros(seg_array.shape)
        
        pred_organ_1 = np.ones(pred_seg.shape)
        target_organ_1 = np.ones(seg_array.shape)

        pred_organ[pred_seg == organ_index] = 1
        target_organ[seg_array == organ_index] = 1
        
        pred_organ_1[pred_seg == organ_index] = 0
        target_organ_1[seg_array == organ_index] = 0

        # 如果该例数据中不存在某一种器官，在表格中记录 None 跳过即可
        if target_organ.sum() == 0:
            worksheet.write(organ_index, file_index + 1, 'None')

        else:
            dice = (2 * pred_organ * target_organ).sum() / (pred_organ.sum() + target_organ.sum())
            worksheet.write(organ_index, file_index + 1, dice)
            PPV= (pred_organ * target_organ).sum() / pred_organ.sum()
            TN = (pred_organ_1 * target_organ_1).sum()
            TP = (pred_organ * target_organ).sum()
            T_P = (pred_organ.sum() + pred_organ_1.sum())
            JAC = (pred_organ * target_organ).sum() / ((pred_organ.sum() + target_organ.sum())-(pred_organ * target_organ).sum())
            ACC = (TN + TP) / T_P
            SPEC = TN / target_organ_1.sum()
            print('dice\t','PPV\t','ACC\t','JAC\t','SPEC\t',)
            print('%.2f'%dice,'\t','%.2f'%PPV,'\t','%.2f'%ACC,'\t','%.2f'%JAC,'\t','%.2f'%SPEC,'\t',)
            worksheet.write(17, file_index + 1, PPV)
            #worksheet.write(16, file_index + 1, ACC)
            worksheet.write(18, file_index + 1, JAC)
            #worksheet.write(19, file_index + 1, SPEC)
            

    # 将预测的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(pred_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(organ_pred_dir, file.replace('img', 'organ')))
    del pred_seg

    speed = time() - start_time

    worksheet.write(14, file_index + 1, speed)

    print('this case use {:.3f} s'.format(speed))
    print('-----------------------')


# 最后安全关闭表格
workbook.close()
