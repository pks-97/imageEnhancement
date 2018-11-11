import cv2
import numpy as np
import matplotlib.pyplot as plt
# from skimage import restoration
# from PIL import Image

# Reading the image and reshaping it
img = cv2.imread('images/6.png',-1)
rows,cols,dim=img.shape
(b,g,r,a) = cv2.split(img)
result = np.zeros((rows,cols,3))
result = cv2.merge([b,g,r])
img = result
img = cv2.resize(img,(512,512))
cv2.imshow('image',img)
cv2.waitKey(0)


# code for homomorphic filter
def homomorphic(img):

	img = np.float32(img)
	img = img/255
	rows,cols,dim=img.shape

	#rh,rl are high frequency and low frequency gain respectively.the cutoff 32 is kept for 512,512 images
	#but it seems to work fine otherwise

	rh, rl, cutoff = 1.3,0.8,32
	b,g,r = cv2.split(img)
	y_log_b = np.log(b+0.01)
	y_log_g = np.log(g+0.01)
	y_log_r = np.log(r+0.01)
	y_fft_b = np.fft.fft2(y_log_b)
	y_fft_g = np.fft.fft2(y_log_g)
	y_fft_r = np.fft.fft2(y_log_r)
	y_fft_shift_b = np.fft.fftshift(y_fft_b)
	y_fft_shift_g = np.fft.fftshift(y_fft_g)
	y_fft_shift_r = np.fft.fftshift(y_fft_r)


	#D0 is the cutoff frequency again a parameter to be chosen

	D0 = cols/cutoff
	H = np.ones((rows,cols))
	B = np.ones((rows,cols))
	for i in range(rows):
		for j in range(cols):
			H[i][j]=((rh-rl)*(1-np.exp(-((i-rows/2)**2+(j-cols/2)**2)/(2*D0**2))))+rl #DoG filter


	result_filter_b = H * y_fft_shift_b
	result_filter_g = H * y_fft_shift_g
	result_filter_r = H * y_fft_shift_r

	result_interm_b = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter_b)))
	result_interm_g = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter_g)))
	result_interm_r = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter_r)))

	result_b = np.exp(result_interm_b)
	result_g = np.exp(result_interm_g)
	result_r = np.exp(result_interm_r)

	result = np.zeros((rows,cols,dim))
	result[:,:,0] = result_b
	result[:,:,1] = result_g
	result[:,:,2] = result_r

	ma=-1
	mi = 500
	for i in range(3):
		r = max(np.ravel(result[:,:,i]))
		x = min(np.ravel(result[:,:,i]))
		if r > ma:
			ma=r
		if x < mi:
			mi = x

	#norm_image = cv2.normalize(result,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	
	return(result)

#code for adaptive histogram equalization   	
def adapt_histogram(img):

	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(2,2))
	img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
	result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	return(result)

###### code contrast adjustment ##### 
def normalizeRed(img):
 
    minI    = min(np.ravel(img))
    maxI    = max(np.ravel(img))
    print(minI,maxI)
    minO    = 0
    maxO    = 255

    for i in range(img.shape[0]):
    	for j in range(img.shape[1]):
    		img[i,j] = (img[i,j]-minI)*(((maxO-minO)/(maxI-minI))+minO)

    return img



def normalizeGreen(img):

    minI    = min(np.ravel(img))
    maxI    = max(np.ravel(img))
    print(minI,maxI)
    minO    = 0
    maxO    = 235

    for i in range(img.shape[0]):
    	for j in range(img.shape[1]):
    		img[i,j] = (img[i,j]-minI)*(((maxO-minO)/(maxI-minI))+minO)

    return img



def normalizeBlue(img):

    minI    = min(np.ravel(img))
    maxI    = max(np.ravel(img))
    print(minI,maxI)
    minO    = 0
    maxO    = 245

    for i in range(img.shape[0]):
    	for j in range(img.shape[1]):
    		img[i,j] = (img[i,j]-minI)*(((maxO-minO)/(maxI-minI))+minO)

    return img


result1 = adapt_histogram(img)
imageObject=result1
b,g,r = cv2.split(imageObject)


normalizedRedBand      = normalizeRed(r)
normalizedGreenBand    = normalizeGreen(g)
normalizedBlueBand     = normalizeBlue(b)

normalizedImage = cv2.merge([normalizedBlueBand,normalizedGreenBand,normalizedRedBand])
normalizedImage = adapt_histogram(normalizedImage)
res = cv2.GaussianBlur(normalizedImage,(11,11),0)

cv2.imshow('after first contrast',res)
cv2.waitKey(0)
# cv2.destroyAllWindows()

result2 = normalizedImage
smooth = result2 
res = homomorphic(smooth)
res = cv2.GaussianBlur(res,(11,11),0)
img0 = res
cv2.imwrite("result/test_homo_1.jpg",res)
cv2.imshow('after contrast',img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
