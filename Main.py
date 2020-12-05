from google.colab import drive
drive.mount('/content/drive')
import utils
import torch,utils,train
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from engine import train_one_epoch, evaluate
import scipy.io
import os
from engine import train_one_epoch, evaluate
import utils

# !pip install numpy==1.17.1 This is necessary. Make sure that you use correct numpy version
import numpy as np

import scipy.io
import os
direction = '/content/drive/My Drive/Colab Notebooks/BrainTumor/AllImagesMatFiles/'
s = 0



class BrainTumor(object):
    def __init__(self, root, transforms):
        self.root, self.transforms = root, transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "AllImagesMatFiles"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "AllMasksMatFiles"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "AllLabelsMatFiles"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "AllImagesMatFiles", self.imgs[idx])
        mask_path = os.path.join(self.root, "AllMasksMatFiles", self.masks[idx])
        label_path = os.path.join(self.root, "AllLabelsMatFiles", self.labels[idx])

        img = np.array(scipy.io.loadmat(img_path, mdict=None, appendmat=True)['img'], dtype=float)
        biggest_value = np.amax(img)
        img = torch.tensor(data=img / biggest_value, dtype=torch.float)
        img = img.reshape(shape=(1, 512, 512))
        mask = np.array(scipy.io.loadmat(mask_path, mdict=None, appendmat=True)['mask'], dtype=int)
        label = np.array(scipy.io.loadmat(label_path, mdict=None, appendmat=True)['label'], dtype=int)
        label = int(label[0])

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        # masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []

        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = boxes.reshape(shape=(1, 4))
        labels = label * torch.ones((num_objs,), dtype=torch.int64)
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.reshape(shape=(1, 512, 512))

        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = mask
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



backbone = torchvision.models.mobilenet_v2(pretrained=True).features

backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                 aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                              output_size=7,
                                                sampling_ratio=2)

mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                     output_size=14,
                                                   sampling_ratio=2)

model = MaskRCNN(backbone,
                num_classes=4,
             rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            mask_roi_pool=mask_roi_pooler)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataSet = BrainTumor('/content/drive/My Drive/Colab Notebooks/BrainTumor/',transforms=None)
data_loader = torch.utils.data.DataLoader(
        dataSet, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader, device=device)


model.eval()
direction = '/content/drive/My Drive/Colab Notebooks/BrainTumor/AllImagesMatFilesTest/'
for zz in os.listdir(direction):
    test1 = np.array(scipy.io.loadmat( direction+zz ,mdict=None,appendmat=True)['img'],dtype=float)
    biggest = np.amax(test1)
    test1 = torch.tensor(test1/biggest,dtype=torch.float)
    test1 = [test1.reshape(shape=(1,512,512)).to(device)]
    predictions = model(test1)
    print(predictions)