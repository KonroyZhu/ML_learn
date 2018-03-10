from skimage.measure import label,regionprops

def segment_image(image):
    labelde_image=label(image >0)#The letters are put in an arry, and when the digit is not 0 it belongs to a lettr
    subimages=[]#To save each letter
    for region in regionprops(labelde_image):
        start_x,start_y,end_x,end_y=region.bbox
        subimages.append(image[start_x:end_x,start_y:end_y])
        if len(subimages)==0:
            return [image,]
        return subimages