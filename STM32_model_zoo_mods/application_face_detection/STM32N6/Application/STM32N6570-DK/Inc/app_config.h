/**
******************************************************************************
* @file    app_config.h
* @author  GPM Application Team
*
******************************************************************************
* @attention
*
* Copyright (c) 2023 STMicroelectronics.
* All rights reserved.
*
* This software is licensed under terms that can be found in the LICENSE file
* in the root directory of this software component.
* If no LICENSE file comes with this software, it is provided AS-IS.
*
******************************************************************************
*/

/* ---------------    Generated code    ----------------- */
#ifndef APP_CONFIG
#define APP_CONFIG

#include "arm_math.h"

#define USE_DCACHE

/*Defines: CMW_MIRRORFLIP_NONE; CMW_MIRRORFLIP_FLIP; CMW_MIRRORFLIP_MIRROR; CMW_MIRRORFLIP_FLIP_MIRROR;*/
#define CAMERA_FLIP CMW_MIRRORFLIP_NONE

#define ASPECT_RATIO_CROP (1) /* Crop both pipes to nn input aspect ratio; Original aspect ratio kept */
#define ASPECT_RATIO_FIT (2) /* Resize both pipe to NN input aspect ratio; Original aspect ratio not kept */
#define ASPECT_RATIO_FULLSCREEN (3) /* Resize camera image to NN input size and display a fullscreen image */
#define ASPECT_RATIO_MODE ASPECT_RATIO_FIT

/* Postprocessing type configuration */
#define POSTPROCESS_TYPE    POSTPROCESS_FD_YUNET_UI

#define COLOR_BGR (0)
#define COLOR_RGB (1)
#define COLOR_MODE COLOR_BGR

/* Postprocessing FD_YUNET configuration */
#define AI_FD_YUNET_PP_NB_KEYPOINTS      (5)
#define AI_FD_YUNET_PP_NB_CLASSES        (1)
#define AI_FD_YUNET_PP_IMG_SIZE          (320)
#define AI_FD_YUNET_PP_OUT_32_NB_BOXES   (100)
#define AI_FD_YUNET_PP_OUT_16_NB_BOXES   (400)
#define AI_FD_YUNET_PP_OUT_8_NB_BOXES    (1600)
#define AI_FD_YUNET_PP_MAX_BOXES_LIMIT   (10)
#define AI_FD_YUNET_PP_CONF_THRESHOLD    (0.5)
#define AI_FD_YUNET_PP_IOU_THRESHOLD     (0.5)
#define WELCOME_MSG_1         "pruned_yunet_structured_int8_20_06.onnx"
#define WELCOME_MSG_2       "Model Running in STM32 MCU internal memory"

#endif      /* APP_CONFIG */
