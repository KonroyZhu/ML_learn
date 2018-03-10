import numpy as np
from skimage import transform as tf
from PIL import  Image ,ImageDraw, ImageFont

def create_captcha(text, shear=0,size=(100,24)):
    im=Image.new("L",size)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("Coval-ExtraLight.otf", 22)
    draw.text((2, -2), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    return image/image.max()


import matplotlib.pyplot  as plt
image=create_captcha("KONROY",shear=0.5)
plt.imshow(image,cmap='Greys')
plt.show()