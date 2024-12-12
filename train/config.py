class Config:
    # 路徑設置
    TRAIN_IMAGE_DIR = "/home/ouvic/ML/ML_Final/dataset_2d_x_test/train/imgs"
    TRAIN_MASK_DIR = "/home/ouvic/ML/ML_Final/dataset_2d_x_test/train/masks"
    VAL_IMAGE_DIR = "/home/ouvic/ML/ML_Final/dataset_2d_x_test/valid/imgs"
    VAL_MASK_DIR = "/home/ouvic/ML/ML_Final/dataset_2d_x/valid/masks"
    INFERENCE_IMAGE_DIR = "/home/ouvic/ML/ML_HW2/dataset/test/imgs"
    OUTPUT_DIR = "/home/ouvic/ML/ML_HW2/predict2/pred_UnetPlusPlus_1212"
    
    CHECKPOINT_PATH = "/home/ouvic/ML/ML_Final/project/pth/UnetPlusPlusImproved_1211.pth"
    # 模型參數
    NUM_CLASSES = 2
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    
    # 數據增強參數
    ORGIN_IMAGE_SIZE = (300, 1259)
    IMAGE_SIZE = (256, 1024)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # 模型配置
    MODEL_NAME = "SimpleUNet"
    ENCODER_NAME = "resnext101_64x4d"
    
    # 損失函數配置
    LOSS_WEIGHT = 0.5