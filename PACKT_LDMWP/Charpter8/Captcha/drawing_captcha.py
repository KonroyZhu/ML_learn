
####drawing image-------------------------------------------
import numpy as np
from skimage import transform as tf
from PIL import  Image ,ImageDraw, ImageFont

file="./bretan/Coval-Black.otf"
# with open(file,mode='r')as f:
#     print(f)


def create_captcha(text, shear=0,size=(100,24)):
    im=Image.new("L",size,"black")
    draw=ImageDraw.Draw(im)
    font=ImageFont.truetype(file,22)
    draw.text((2,-2),text,fill=1,font=font)
    image=np.array(im)
    affine_tf=tf.AffineTransform(shear=shear)
    image=tf.warp(image,affine_tf)
    return image

####To save the array in file_image making it easier to understand
image_file = "./image"
with open(image_file,mode='w') as f:
   image=create_captcha("Konroy",shear=0.5)
   for i in image:
        f.write('\n')
        for content in i:
            str_content=str(content)
            # print(len(str_content))
            if len(str_content)>3:
                str_content='1.0'
            print(str_content)
            f.write(str_content+' ')

####segment_image-------------------------------------------

from skimage.measure import label,regionprops

# def segment_image(image):
#     labelde_image=label(image >0)#The letters are put in an arry, and when the digit is not 0 it belongs to a lettr
#     subimages=[]#To save each letter
#     for region in regionprops(labelde_image):
#         start_x,start_y,end_x,end_y=region.bbox
#         subimages.append(image[start_x:end_x,start_y:end_y])
#         if len(subimages)==0:
#             return [image,]
#         return subimages
#
# subimages=segment_image(image)
# f,axes=plt.subplots(1,len(subimages))
# for i in range(len(subimages)):
#     plt.imshow(subimages[i])
# plt.show()