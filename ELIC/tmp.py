import cv2

# 读取图像
image = cv2.imread('images/23394.png')

# 对图像进行边缘保留滤波
filtered_image = cv2.bilateralFilter(image, d=2, sigmaColor=85, sigmaSpace=5)

# 显示原始图像和滤波后的图像
cv2.imwrite('Original_Image.png', image)
cv2.imwrite('Filtered_Image.png', filtered_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import os
import numpy as np

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_val = 255.0
    return 20 * np.log10(max_val / np.sqrt(mse))

# 读取文件A中的全部图像
folder_A = './images/inputs'
folder_B = './images/BFIlter'
if not os.path.exists(folder_B):
    os.makedirs(folder_B)

for filename in os.listdir(folder_A):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_A = cv2.imread(os.path.join(folder_A, filename))


        if 0:
            # 对图像应用双边滤波器进行滤波
            # 对图像进行边缘保留滤波
            img_B = cv2.bilateralFilter(img_A, d=2, sigmaColor=55, sigmaSpace=1)

            # 将滤波后的图像保存到文件B中
            cv2.imwrite(os.path.join(folder_B, filename), img_B)

        else:
            img_B = cv2.imread(folder_B+'/rec_23211.png')
            img_B = cv2.imread(folder_B+'/rec_sm_23211.png')
            # img_B = cv2.imread(folder_B+'/23211.png')

        # 计算PSNR
        psnr_val = psnr(img_A, img_B)
        print(f'PSNR for {filename}: {psnr_val}')

img1 = cv2
