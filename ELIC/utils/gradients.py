import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters


def GM_caculate( im):
  sigma = 0.5  # 0.5
  # im = np.squeeze(im, axis=1).cpu()

  # N = im.size()[0]
  # Gm_list = []
  # img = np.squeeze(im[0, ::])
  imgx = np.zeros(im.shape)
  imgy = np.zeros(im.shape)

  # for i in range(0, N):
  filters.gaussian_filter(im, sigma, (0, 1), imgx)
  filters.gaussian_filter(im, sigma, (1, 0), imgy)
  GM = np.sqrt(imgx ** 2 + imgy ** 2)
    # Gm_list.append(GM)
  #
  # Ix_ave = torch.filter2D(imgx.astype('float32'), -1, kernel_3x3, borderType=cv2.BORDER_CONSTANT)
  #
  # # print("Ix_ave = ", imgx)
  # Iy_ave = cv2.filter2D(imgy.astype('float32'), -1, kernel_3x3, borderType=cv2.BORDER_CONSTANT)
  #
  #
  # Ang_ave = np.arctan(np.divide(Ix_ave , Iy_ave + eps))
  # Ang_ave[Iy_ave == 0] = np.pi / 2

  # RO = np.arctan(np.divide(imgx ,  imgy+eps))
  #
  # RO[imgy == 0]= np.pi / 2
  # RO = RO - Ang_ave
  # RM = torch.sqrt(np.multiply((imgx - Ix_ave), (imgx - Ix_ave)) + np.multiply((imgx - Iy_ave), (imgx - Iy_ave)))
  return GM  # , RO, RM

# 读取图像
image_path = './../images/desert_2.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_GM = GM_caculate(image)
img_GM = (img_GM - img_GM.min())/(img_GM.max() - img_GM.min())

# cv2.imshow("GM_orin", img_GM)

# 计算图像的梯度
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向梯度
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向梯度

# 计算梯度的幅值和方向
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
gradient_direction = np.arctan2(sobely, sobelx)

# 显示原始图像
plt.figure(1)
plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# 显示水平方向梯度
plt.subplot(2, 2, 2), plt.imshow(np.abs(sobelx), cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

# 显示垂直方向梯度
plt.subplot(2, 2, 3), plt.imshow(np.abs(sobely), cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

# 显示梯度幅值
plt.figure(4)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude'), plt.xticks([]), plt.yticks([])

plt.show()
# cv2.waitKey(0)
