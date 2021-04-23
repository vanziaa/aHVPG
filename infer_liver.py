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


organ_pred_dir = './Path to save segmentation/'

module_dir = './FINAL_liver_net.pth'

upper = 350
lower = -upper
down_scale = 0.5
size = 48
slice_thickness = 3


organ_list = [
    'LIVER',

]



net = torch.nn.DataParallel(Net(training=False)).cuda()
net.load_state_dict(torch.load(module_dir))
net.eval()



for file_index, file in enumerate(os.listdir(val_ct_dir)):

    start_time = time()

    
    ct = sitk.ReadImage(os.path.join(val_ct_dir, file), sitk.sitkInt16)
    ct_array_ori = sitk.GetArrayFromImage(ct)
    ct_array= sitk.GetArrayFromImage(ct)
    print(file)
    print('size of CT: ', ct_array.shape)

    
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / slice_thickness, down_scale, down_scale), order=3)
    #ct_array = ndimage.shift(ct_array,shift=[0,360,0],mode='reflect')

    
    flag = False
    start_slice = 0
    end_slice = start_slice + size - 1
    ct_array_list = []

    while end_slice <= ct_array.shape[0] - 1:
        ct_array_list.append(ct_array[start_slice:end_slice + 1, :, :])

        start_slice = end_slice + 1
        end_slice = start_slice + size - 1

    
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

            
            outputs_list.append(outputs.cpu().detach().numpy())
            del outputs

    pred_seg = np.concatenate(outputs_list[0:-1], axis=1)
    if flag is False:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=1)
    else:
        pred_seg = np.concatenate([pred_seg, outputs_list[-1][:, -count:, :, :]], axis=1)


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



#workbook.close()
