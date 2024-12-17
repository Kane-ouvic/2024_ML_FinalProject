# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:57:43 2024

@author: user
"""
class Config:
    # 訓練參數
    IN_CHANNEL = 1
    NUM_CLASSES = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    REDUCT_STEP = 10
    REDUCT_FACTOR = 0.5
    BATCH_SIZE = 2
    ACCUMULATION_STEP = 4
    
    # 模型配置
    MODEL_NAME = 'model_name'
    ENCODER_NAME = 'resnet50'
    PRETRAIN_WEIGHT = 'imagenet'
    ENCODER_DEPTH = 5
    ENCODER_CHANNEL = (512, 256, 128, 64, 32)
    ATTENTION_TYP = None