# Import necessary libraries#---------------------------------------------------#
import matplotlib.pyplot as plt
import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
# cv2.setUseOptimized(False)
# cv2.setNumThreads(0)
import torchvision.models.segmentation
import torch
import os
from tqdm import tqdm
import time
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn as MaskRCNN_ResNet50_FPN_Weights



#---------------------------------------------------#



#-----------------------------------------------------------------------------------------------------#
# Check if GPU is available#---------------------------------------------------#
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)
#ajouter un msg de si gpu si cpu
#-----------------------------------------------------------------------------------------------------#

# Directories of the images and their masks#---------------------------------------------------#
imageDir = "/media/dd2To/Amine_Mask/DataImgs/EndoImages"
maskDir = "/media/dd2To/Amine_Mask/DataImgs/EndoMasks"
#-----------------------------------------------------------------------------------------------------#

# Function that loads image names from a directory


def loadImageNames(imageDir):
    return [f for f in os.listdir(imageDir) if os.path.isfile(os.path.join(imageDir, f))]


# Loading image names for training
imgs = loadImageNames(imageDir)
print("Number of training images:", len(imgs))


# #-----------------------------------------------------------------------------------------------------#

# Desired image size and batch size for the DataLoader
imageSize = (1280, 960)
batchSize = 2
# # #-----------------------------------------------------------------------------------------------------#

# path to save models after training
save_dir = "/media/dd2To/model_saves"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def loadTrainingData():
    # Initialize empty lists for images and data n
    batch_Imgs = []
    batch_Data = []
    while len(batch_Imgs) < batchSize and len(imgs) > 0:
        # Choose a random image
        idx = random.randint(0, len(imgs)-1)
        imgName = imgs[idx]  

        # Load and resize image
        img = cv2.imread(os.path.join(imageDir, imgName))
        if img is None:
            print(f"Image '{imgName}' could not be read")
            continue  # Skip this image and try another one

        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)

        # Load corresponding mask
        mask_path = os.path.join(maskDir, imgName.replace('.png', '_mask.png'))
        masks = []

        # If mask exists, load and resize it
        if os.path.exists(mask_path):
            vesMask = cv2.imread(mask_path, 0)
            vesMask = (vesMask > 0).astype(np.uint8)
            vesMask = cv2.resize(vesMask, imageSize, cv2.INTER_NEAREST)
            masks.append(vesMask)

        # Check the number of objects in the image
        num_objs = len(masks)
        if num_objs == 0:
            continue  # Skip this image and try another one

        # Initialize boxes tensor
        boxes = torch.zeros([num_objs, 4], dtype=torch.float32)
        for i in range(num_objs):
            x, y, w, h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])

        # Convert masks and image to tensors
        masks = np.array(masks)  # convert list to a numpy array
        masks = torch.from_numpy(masks)  # convert numpy array to tensor
        img = torch.as_tensor(img, dtype=torch.float32)

        # Prepare data dictionary
        data = {}
        data["boxes"] = boxes
        data["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        data["masks"] = masks

        # Append to batch lists
        batch_Imgs.append(img)
        batch_Data.append(data)

    # Convert to tensor and adjust dimensions
    batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
    batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
    return batch_Imgs, batch_Data


# # # #---------------------------------TEST/VALIDATION--------------------------------------------------------------------#
# # Chemins des données de validation
# #
# val_image_dir = "/home/etudiant_master/Amine_Mask/DataImgs/test_endoimgs"
# #
# val_mask_dir = "/home/etudiant_master/Amine_Mask/DataImgs/data_augmentation.zip"
# imgs_test= loadImageNames(val_image_dir)


# def loadValidationData():
#     # Initialize empty lists for images and data
#     batch_Imgs = []
#     batch_Data = []
#     while len(batch_Imgs) < batchSize and len(imgs_test) > 0:
#         # Choose a random image
#         idx = random.randint(0, len(imgs_test)-1)
#         imgName_test = imgs_test[idx]

#         # Load and resize image
#         img_test = cv2.imread(os.path.join(val_image_dir, imgName_test))
#         if img_test is None:
#             print(f"Image '{imgName_test}' could not be read")
#             continue  # Skip this image and try another one

#         img_test = cv2.resize(img_test, imageSize, cv2.INTER_LINEAR)

#         # Load corresponding mask
#         mask_path_test = os.path.join(
#             val_mask_dir, imgName_test.replace('.png', '_mask.png'))
#         masks_test = []

#         # If mask exists, load and resize it
#         if os.path.exists(mask_path_test):
#             vesMask_test = cv2.imread(mask_path_test, 0)
#             vesMask_test = (vesMask_test > 0).astype(np.uint8)
#             vesMask_test = cv2.resize(
#                 vesMask_test, imageSize, cv2.INTER_NEAREST)
#             masks_test.append(vesMask_test)

#         # Check the number of objects in the image
#         num_objs = len(masks_test)
#         if num_objs == 0:
#             continue  # Skip this image and try another one

#         # Initialize boxes tensor
#         boxes_test = torch.zeros([num_objs, 4], dtype=torch.float32)
#         for i in range(num_objs):
#             x, y, w, h = cv2.boundingRect(masks_test[i])
#             boxes_test[i] = torch.tensor([x, y, x+w, y+h])

#         # Convert masks and image to tensors
#         masks_test = np.array(masks_test)  # convert list to a numpy array
#         # convert numpy array to tensor
#         masks_test = torch.from_numpy(masks_test)
#         img_test = torch.as_tensor(img_test, dtype=torch.float32)

#         # Prepare data dictionary
#         data = {}
#         data["boxes"] = boxes_test
#         data["labels"] = torch.ones((num_objs,), dtype=torch.int64)
#         data["masks"] = masks_test

#         # Append to batch lists
#         batch_Imgs.append(img_test)
#         batch_Data.append(data)

#     # Convert to tensor and adjust dimensions
#     batch_Imgs = torch.stack([torch.as_tensor(d) for d in batch_Imgs], 0)
#     batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
#     return batch_Imgs, batch_Data



# #--------------------------------------------------------------------------------------------------------------------------------------------------#


# Initialize the model
model = maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
model.to(device)
# # #-----------------------------------------------------------------------------------------------------#

# Set up the optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
# # #-----------------------------------------------------------------------------------------------------#

# Switch the model to train mode
model.train()
# # #-----------------------------------------------------------------------------------------------------#

num_epochs = 100


# Initialisation des listes pour garder une trace des pertes
train_losses = []
val_losses = []

# # Function for validation


# def validate(model, val_image_dir, val_mask_dir):
#     # Load validation image names
#     val_imgs = loadImageNames(val_image_dir)

#     # Switch model to evaluation mode
#     model.eval()
#     model.to(device)

#     total_val_loss = 0
#     with torch.no_grad():
#         for idx, img_name in enumerate(val_imgs):
#             img_test = cv2.imread(os.path.join(val_image_dir, img_name))
#             if img_test is None:
#                 continue
#             img_test = cv2.resize(img_test, imageSize, cv2.INTER_LINEAR)
#             img_test = torch.as_tensor(
#                 img_test, dtype=torch.float32).to(device)

#             mask_path = os.path.join(
#                 val_mask_dir, img_name.replace('.png', '_mask.png'))
#             if not os.path.exists(mask_path):
#                 continue
#             vesMask = cv2.imread(mask_path, 0)
#             vesMask = (vesMask > 0).astype(np.uint8)
#             vesMask = cv2.resize(vesMask, imageSize, cv2.INTER_NEAREST)
#             masks_test = torch.from_numpy(np.array([vesMask])).to(device)

#             # Add a batch dimension
#             img_test = img_test.unsqueeze(0)
#             masks_test = masks_test.unsqueeze(0)

#             # Prepare target dictionary
#             target = {}
#             target["boxes"] = torch.empty(
#                 (0, 4), dtype=torch.float32, device=device)
#             target["labels"] = torch.empty(
#                 (0,), dtype=torch.int64, device=device)
#             target["masks"] = masks_test

#             # Compute loss
#             loss_dict = model([img_test], [target])
#             val_loss = sum(loss for loss in loss_dict.values())
#             total_val_loss += val_loss.item()

#     # Switch model back to training mode
#     model.train()
#     model.to(device)

#     # Calculate average validation loss
#     avg_val_loss = total_val_loss / len(val_imgs)
#     return avg_val_loss





# Boucle d'entraînement#--------------------------------------------------------------------------------------""
for i in tqdm(range(num_epochs), desc="Training progress", unit="epoch"):
    # Load data
    images, targets = loadTrainingData()
    total_images = len(imgs)
    processed_images = 0

    # Move data to the target device
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Clear the gradients of the optimized variables
    optimizer.zero_grad()

    # Forward pass: compute predicted outputs by passing inputs to the model
    loss_dict = model(images, targets)

    # Calculate the batch loss
    losses = sum(loss for loss in loss_dict.values())
    # Append the training loss for this epoch
    train_losses.append(losses.item())

    # Backward pass: compute gradient of the loss with respect to model parameters
    losses.backward()

    # Perform a single optimization step (parameter update)
    optimizer.step()




    #-------------------------------------------------------------------------------------------------------#

    # Log progress
    print(i, 'loss:', losses.item())
    if (i + 1) % 1 == 0:  # You can adjust the frequency of saving here (e.g., every 1 epoch)
        save_path = os.path.join(save_dir, f"E6_gpu_new_jadid_model_after_{i + 1}_epochs.torch")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved after {i + 1} epochs.")

        # # Perform validation
        # val_loss = validate(model, val_image_dir, val_mask_dir)
        # # Append the validation loss for this epoch
        # val_losses.append(val_loss)
        # print(f'Validation loss after {i + 1} epochs: {val_loss}')

# Après l'entraînement, tracez la perte d'entraînement et de validation au fil du temps

# epochs = [epoch for epoch in range(1, num_epochs + 1)]
# plt.figure(figsize=(10, 5))
# plt.plot(epochs, train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
# plt.plot(epochs, val_losses, label='Validation Loss', color='red', linestyle='--', marker='x')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Courbes')
# plt.legend()
# plt.show()





print(train_losses)