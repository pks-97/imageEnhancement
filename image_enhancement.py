import cv2
import numpy as np 


im = cv2.imread("images/10.jpg")
# cv2.imshow("test",im)
# print(im[0][0])


image = im

m = [-200,-200,-200]
mi = [1000,1000,1000]
[rows,cols,dim] = im.shape
# print(rows)
# print(cols)
# arr = np.array(im)
# print(arr)
for k in range(0,3):

    for i in range(0,rows):
	    for j in range(0,cols):
		    # intensity = im[i][j][0]/3 + im[i][j][1]/3 + im[i][j][2]/3
		    intensity = im[i][j][k]
		    if intensity  > m[k]:
			    m[k] = intensity

		    if intensity < mi[k]:
		        mi[k] = intensity


for k in range(0,3):
	for i in range(0,rows):
		for j in range(0,cols):
			intensity = im[i][j][k]
			im[i][j][k] = (((intensity - mi[k])*(255 - mi[k]))/(m[k] - mi[k])) + 0


# hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)

# print(hsv[0][0][0])

# m1 = [-200,-200]
# mi1 = [1000,1000]

# for k in range(0,2):

#     for i in range(0,rows):
# 	    for j in range(0,cols):
# 		    # intensity = im[i][j][0]/3 + im[i][j][1]/3 + im[i][j][2]/3
# 		    intensity = hsv[i][j][k+1]
# 		    if intensity  > m[k]:
# 			    m1[k] = intensity

# 		    if intensity < mi[k]:
# 		        mi1[k] = intensity


# for k in range(0,2):
# 	for i in range(0,rows):
# 		for j in range(0,cols):
# 			intensity = hsv[i][j][k+1]
# 			hsv[i][j][k+1] = (((intensity - mi1[k])*(255 - mi1[k]))/(m1[k] - mi1[k])) + 0


# test = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
# cv2.imwrite("test_final_2.png",im)
cv2.imshow("test",im)
cv2.waitKey(0)
cv2.destroyAllWindows()