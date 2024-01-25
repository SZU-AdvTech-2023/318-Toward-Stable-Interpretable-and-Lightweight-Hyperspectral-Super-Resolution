import scipy.io as scio
import cv2
import numpy as np
import torch
import tifffile as tiff
import spectral as spy
import matplotlib.pyplot as plt

# mat = scio.loadmat('/home1/mzq/pycode/DAEM2-master/results/16.mat')
# data = mat['data']

# data = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(dim=0)
# data = data.float().detach()
# # data = data.detach().cpu().numpy()

# data = np.squeeze(data.detach().cpu().numpy())
# data_red = data[30, :, :][:, :, np.newaxis]
# data_green = data[16, :, :][:, :, np.newaxis]
# data_blue = data[5, :, :][:, :, np.newaxis]
# data = np.concatenate((data_blue, data_green, data_red), axis=2)
# data = 255*(data-np.min(data))/(np.max(data)-np.min(data))
# # lr = cv2.resize(data, (data.shape[2], data.shape[3]), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite('./figs/test_16.jpg', data)




# mat1 = scio.loadmat('/home1/mzq/pycode/DAEM2-master/CAVE/gt/16.mat')
# data1 = mat1['data']

# data1 = torch.from_numpy(data1).permute(2, 0, 1).unsqueeze(dim=0)
# data1 = data1.float().detach()
# # data = data.detach().cpu().numpy()

# data1 = np.squeeze(data1.detach().cpu().numpy())
# data_red1 = data1[30, :, :][:, :, np.newaxis]
# data_green1 = data1[16, :, :][:, :, np.newaxis]
# data_blue1 = data1[5, :, :][:, :, np.newaxis]
# data1 = np.concatenate((data_blue1, data_green1, data_red1), axis=2)
# data1 = 255*(data1-np.min(data1))/(np.max(data1)-np.min(data1))
# # data1 = cv2.resize(data1, (data1.shape[1], data1.shape[2]), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite('./figs/test_16_gt.jpg', data1)



# result
mat = scio.loadmat('/home1/mzq/pycode/DAEM2-master/results/19.mat')
data = mat['data']

data = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(dim=0)
data = data.float().detach()
# data = data.detach().cpu().numpy()

data = np.squeeze(data.detach().cpu().numpy())
data = data[15, :, :][:, :, np.newaxis]
data = 255*(data-np.min(data))/(np.max(data)-np.min(data))
# lr = cv2.resize(data, (data.shape[2], data.shape[3]), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite('./figs/test_result_19.png', data)

mat = scio.loadmat('/home1/mzq/pycode/DAEM2-master/figs/19_test.mat')
data = mat['data']

data = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(dim=0)
data = data.float().detach()
# data = data.detach().cpu().numpy()

data = np.squeeze(data.detach().cpu().numpy())
data = data[15, :, :][:, :, np.newaxis]
data = 255*(data-np.min(data))/(np.max(data)-np.min(data))
# lr = cv2.resize(data, (data.shape[2], data.shape[3]), interpolation=cv2.INTER_NEAREST)
cv2.imwrite('./figs/improve_19.png', data)



# gt
mat = scio.loadmat('/home1/mzq/pycode/DAEM2-master/CAVE/gt/19.mat')
data = mat['data']

data = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(dim=0)
data = data.float().detach()
# data = data.detach().cpu().numpy()

data = np.squeeze(data.detach().cpu().numpy())
data = data[15, :, :][:, :, np.newaxis]
data = 255*(data-np.min(data))/(np.max(data)-np.min(data))
# lr = cv2.resize(data, (data.shape[2], data.shape[3]), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite('./figs/test_gt_19.png', data)



# hsi
mat = scio.loadmat('/home1/mzq/pycode/DAEM2-master/CAVE/hsi_5/19.mat')
data = mat['data']

data = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(dim=0)
data = data.float().detach()
# data = data.detach().cpu().numpy()

data = np.squeeze(data.detach().cpu().numpy())
data = data[15, :, :][:, :, np.newaxis]
data = 255*(data-np.min(data))/(np.max(data)-np.min(data))
data = cv2.resize(data, (512, 512), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite('./figs/test_hsi_19.png', data)




# msi
mat = scio.loadmat('/home1/mzq/pycode/DAEM2-master/CAVE/msi/19.mat')
data = mat['data']

data = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(dim=0)
data = data.float().detach()
# data = data.detach().cpu().numpy()

data = np.squeeze(data.detach().cpu().numpy())
data = data[1, :, :][:, :, np.newaxis]
data = 255*(data-np.min(data))/(np.max(data)-np.min(data))
# data = cv2.resize(data, (512, 512), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite('./figs/test_msi_19.png', data)



# 加载融合前和融合后的高光谱图像
image1 = cv2.imread('/home1/mzq/pycode/DAEM2-master/figs/improve_19.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('/home1/mzq/pycode/DAEM2-master/figs/test_gt_19.png', cv2.IMREAD_GRAYSCALE)

difference_image = cv2.absdiff(image1, image2)

# enhanced_image = cv2.equalizeHist(difference_image)

pseudo_colored_image = cv2.applyColorMap(difference_image, cv2.COLORMAP_JET)

# plt.imshow(cv2.cvtColor(pseudo_colored_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()


# 计算MSE误差图
# error = (pre_fusion_image - post_fusion_image) ** 2

# # 计算均方误差
# mse_error = np.mean(error)

# # 标准化误差值，确保分母不为零
# if mse_error != 0:
#     normalized_error = ((mse_error - error) / mse_error * 255).astype(np.uint8)
# else:
#     normalized_error = np.zeros_like(error, dtype=np.uint8)

# # 创建伪彩色图像
# error_map_color = cv2.applyColorMap((normalized_error * 255).astype(np.uint8), cv2.COLORMAP_JET)

# # 可视化误差图
# cv2.imshow('Error Map', error_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



cv2.imwrite('./figs/improve_19_gt_error.jpg', pseudo_colored_image)

# spy.view_cube(data, bands=[30, 16, 5])



