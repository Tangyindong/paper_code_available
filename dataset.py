import cv2
import random
import math
import numpy as np
from PIL import Image
import yaml
from utils.dataset import Dataset
from utils.utils import non_max_suppression

class ShapesDataset(Dataset):
    #得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n
    #解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self,image_id):
        info=self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp=yaml.load(f.read(), Loader=yaml.FullLoader)
            labels=temp['label_names']
            del labels[0]
        return labels
    #重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(np.shape(mask)[1]):
                for j in range(np.shape(mask)[0]):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] =1
                        # cv2.imshow("mask",mask)
        return mask

    #并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_shapes(self, count, img_floder, mask_floder, imglist, yaml_floder):
        self.add_class("shapes", 1, "rock01")
        self.add_class("shapes", 2, "rock02")
        self.add_class("shapes", 3, "rock03")
        self.add_class("shapes", 4, "rock04")
        self.add_class("shapes", 5, "rock05")
        self.add_class("shapes", 6, "rock06")
        self.add_class("shapes", 7, "rock07")
        self.add_class("shapes", 8, "rock08")
        self.add_class("shapes", 9, "rock09")
        self.add_class("shapes", 10, "rock10")
        self.add_class("shapes", 11, "rock11")
        self.add_class("shapes", 12, "rock12")
        self.add_class("shapes", 13, "rock13")
        self.add_class("shapes", 14, "rock14")
        self.add_class("shapes", 15, "rock15")
        self.add_class("shapes", 16, "rock16")
        self.add_class("shapes", 17, "rock17")
        self.add_class("shapes", 18, "rock18")
        self.add_class("shapes", 19, "rock19")
        self.add_class("shapes", 20, "rock20")
        self.add_class("shapes", 21, "rock21")
        self.add_class("shapes", 22, "rock22")
        self.add_class("shapes", 23, "rock23")
        self.add_class("shapes", 24, "rock24")
        self.add_class("shapes", 25, "rock25")
        self.add_class("shapes", 26, "rock26")
        self.add_class("shapes", 27, "rock27")
        self.add_class("shapes", 28, "rock28")
        self.add_class("shapes", 29, "rock29")
        self.add_class("shapes", 30, "rock30")
        self.add_class("shapes", 31, "rock31")
        self.add_class("shapes", 32, "rock32")
        self.add_class("shapes", 33, "rock33")
        self.add_class("shapes", 34, "rock34")
        self.add_class("shapes", 35, "rock35")
        self.add_class("shapes", 36, "rock36")
        self.add_class("shapes", 37, "rock37")
        self.add_class("shapes", 38, "rock38")
        self.add_class("shapes", 39, "rock39")
        self.add_class("shapes", 40, "rock40")
        for i in range(count):
            img = imglist[i]
            if img.endswith(".jpg"):
                img_name = img.split(".")[0]
                img_path = img_floder + img
                mask_path = mask_floder + img_name + ".png"
                yaml_path = yaml_floder + img_name + ".yaml"
                self.add_image("shapes", image_id=i, path=img_path, mask_path=mask_path,yaml_path=yaml_path)
    #重写load_mask
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([np.shape(img)[0], np.shape(img)[1], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        labels=[]
        labels=self.from_yaml_get_class(image_id)
        labels_form=[]
        for i in range(len(labels)):
            if labels[i].find("rock01")!=-1:
                labels_form.append("rock01")
            elif labels[i].find("rock02")!=-1:
                labels_form.append("rock02")
            elif labels[i].find("rock03")!=-1:
                labels_form.append("rock03")
            elif labels[i].find("rock04")!=-1:
                labels_form.append("rock04")
            elif labels[i].find("rock05")!=-1:
                labels_form.append("rock05")
            elif labels[i].find("rock06")!=-1:
                labels_form.append("rock06")
            elif labels[i].find("rock07")!=-1:
                labels_form.append("rock07")
            elif labels[i].find("rock08")!=-1:
                labels_form.append("rock08")
            elif labels[i].find("rock09")!=-1:
                labels_form.append("rock09")
            elif labels[i].find("rock10")!=-1:
                labels_form.append("rock10")
            elif labels[i].find("rock11")!=-1:
                labels_form.append("rock11")
            elif labels[i].find("rock12")!=-1:
                labels_form.append("rock12")
            elif labels[i].find("rock13")!=-1:
                labels_form.append("rock13")
            elif labels[i].find("rock14")!=-1:
                labels_form.append("rock14")
            elif labels[i].find("rock15")!=-1:
                labels_form.append("rock15")
            elif labels[i].find("rock16")!=-1:
                labels_form.append("rock16")
            elif labels[i].find("rock17")!=-1:
                labels_form.append("rock17")
            elif labels[i].find("rock18")!=-1:
                labels_form.append("rock18")
            elif labels[i].find("rock19")!=-1:
                labels_form.append("rock19")
            elif labels[i].find("rock20")!=-1:
                labels_form.append("rock20")
            elif labels[i].find("rock21")!=-1:
                labels_form.append("rock21")
            elif labels[i].find("rock22")!=-1:
                labels_form.append("rock22")
            elif labels[i].find("rock23")!=-1:
                labels_form.append("rock23")
            elif labels[i].find("rock24")!=-1:
                labels_form.append("rock24")
            elif labels[i].find("rock25")!=-1:
                labels_form.append("rock25")
            elif labels[i].find("rock26")!=-1:
                labels_form.append("rock26")
            elif labels[i].find("rock27")!=-1:
                labels_form.append("rock27")
            elif labels[i].find("rock28")!=-1:
                labels_form.append("rock28")  
            elif labels[i].find("rock29")!=-1:
                labels_form.append("rock29") 
            elif labels[i].find("rock30")!=-1:
                labels_form.append("rock30") 
            elif labels[i].find("rock31")!=-1:
                labels_form.append("rock31") 
            elif labels[i].find("rock32")!=-1:
                labels_form.append("rock32") 
            elif labels[i].find("rock33")!=-1:
                labels_form.append("rock33") 
            elif labels[i].find("rock34")!=-1:
                labels_form.append("rock34") 
            elif labels[i].find("rock35")!=-1:
                labels_form.append("rock35") 
            elif labels[i].find("rock36")!=-1:
                labels_form.append("rock36") 
            elif labels[i].find("rock37")!=-1:
                labels_form.append("rock37") 
            elif labels[i].find("rock38")!=-1:
                labels_form.append("rock38") 
            elif labels[i].find("rock39")!=-1:
                labels_form.append("rock39")  
            elif labels[i].find("rock40")!=-1:
                labels_form.append("rock40")      
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        # cv2.imshow("mask",mask)
        return mask, class_ids.astype(np.int32)
