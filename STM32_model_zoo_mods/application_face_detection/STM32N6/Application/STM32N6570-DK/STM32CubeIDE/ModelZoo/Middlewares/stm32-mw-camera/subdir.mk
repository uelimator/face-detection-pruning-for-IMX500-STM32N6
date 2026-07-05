################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
C:/FD/stm32ai-modelzoo-services/application_code/face_detection/STM32N6/Middlewares/stm32-mw-camera/cmw_camera.c \
C:/FD/stm32ai-modelzoo-services/application_code/face_detection/STM32N6/Middlewares/stm32-mw-camera/cmw_utils.c 

OBJS += \
./Middlewares/stm32-mw-camera/cmw_camera.o \
./Middlewares/stm32-mw-camera/cmw_utils.o 

C_DEPS += \
./Middlewares/stm32-mw-camera/cmw_camera.d \
./Middlewares/stm32-mw-camera/cmw_utils.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/stm32-mw-camera/cmw_camera.o: C:/FD/stm32ai-modelzoo-services/application_code/face_detection/STM32N6/Middlewares/stm32-mw-camera/cmw_camera.c Middlewares/stm32-mw-camera/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DSTM32N657xx '-DECBLOB_CONST_SECTION=__attribute__((section(".flash_blob")))' '-DPOSTPROCESS_WRAPPER_SECTION=__attribute__((section(".psram_bss")))' -DUSE_FULL_ASSERT -DUSE_FULL_LL_DRIVER -DVECT_TAB_SRAM -DLL_ATON_DUMP_DEBUG_API -DLL_ATON_PLATFORM=LL_ATON_PLAT_STM32N6 -DLL_ATON_OSAL=LL_ATON_OSAL_BARE_METAL -DLL_ATON_RT_MODE=LL_ATON_RT_ASYNC -DLL_ATON_SW_FALLBACK -DLL_ATON_DBG_BUFFER_INFO_EXCLUDED=1 -c -I../../Inc -I../../../../Middlewares/ai-postprocessing-wrapper -I../../../../Middlewares/stm32-vision-models-postprocessing/lib_vision_models_pp/Inc -I../../../../Middlewares/stedgeai-lib/Npu/ll_aton -I../../../../Middlewares/stedgeai-lib/Npu/Devices/STM32N6xx -I../../../../Model/STM32N6570-DK -I../../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc -I../../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc/Legacy -I../../../../STM32Cube_FW_N6/Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../../STM32Cube_FW_N6/Drivers/CMSIS/Include -I../../../../STM32Cube_FW_N6/Drivers/CMSIS/DSP/Include -I../../../../STM32Cube_FW_N6/Drivers/BSP/Components/Common -I../../../../STM32Cube_FW_N6/Drivers/BSP/STM32N6570-DK -I../../../../Middlewares/stm32-mw-camera/ISP_Library/isp/Inc -I../../../../Middlewares/stedgeai-lib/Inc -I../../../../STM32Cube_FW_N6/Utilities/lcd -I../../../../Middlewares/stm32-mw-camera -I../../../../STM32Cube_FW_N6/Drivers/BSP/Components/aps256xx -I../../../../Middlewares/stm32-mw-camera/sensors -I../../../../Middlewares/stm32-mw-camera/sensors/imx335 -I../../../../Middlewares/stm32-mw-camera/sensors/vd6g -I../../../../Middlewares/stm32-mw-camera/sensors/vd55g1 -I../../../../Middlewares/stm32-mw-camera/sensors/vd1943 -I../../../../Middlewares/stm32-mw-camera/sensors/ov5640 -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"
Middlewares/stm32-mw-camera/cmw_utils.o: C:/FD/stm32ai-modelzoo-services/application_code/face_detection/STM32N6/Middlewares/stm32-mw-camera/cmw_utils.c Middlewares/stm32-mw-camera/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DSTM32N657xx '-DECBLOB_CONST_SECTION=__attribute__((section(".flash_blob")))' '-DPOSTPROCESS_WRAPPER_SECTION=__attribute__((section(".psram_bss")))' -DUSE_FULL_ASSERT -DUSE_FULL_LL_DRIVER -DVECT_TAB_SRAM -DLL_ATON_DUMP_DEBUG_API -DLL_ATON_PLATFORM=LL_ATON_PLAT_STM32N6 -DLL_ATON_OSAL=LL_ATON_OSAL_BARE_METAL -DLL_ATON_RT_MODE=LL_ATON_RT_ASYNC -DLL_ATON_SW_FALLBACK -DLL_ATON_DBG_BUFFER_INFO_EXCLUDED=1 -c -I../../Inc -I../../../../Middlewares/ai-postprocessing-wrapper -I../../../../Middlewares/stm32-vision-models-postprocessing/lib_vision_models_pp/Inc -I../../../../Middlewares/stedgeai-lib/Npu/ll_aton -I../../../../Middlewares/stedgeai-lib/Npu/Devices/STM32N6xx -I../../../../Model/STM32N6570-DK -I../../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc -I../../../../STM32Cube_FW_N6/Drivers/STM32N6xx_HAL_Driver/Inc/Legacy -I../../../../STM32Cube_FW_N6/Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../../STM32Cube_FW_N6/Drivers/CMSIS/Include -I../../../../STM32Cube_FW_N6/Drivers/CMSIS/DSP/Include -I../../../../STM32Cube_FW_N6/Drivers/BSP/Components/Common -I../../../../STM32Cube_FW_N6/Drivers/BSP/STM32N6570-DK -I../../../../Middlewares/stm32-mw-camera/ISP_Library/isp/Inc -I../../../../Middlewares/stedgeai-lib/Inc -I../../../../STM32Cube_FW_N6/Utilities/lcd -I../../../../Middlewares/stm32-mw-camera -I../../../../STM32Cube_FW_N6/Drivers/BSP/Components/aps256xx -I../../../../Middlewares/stm32-mw-camera/sensors -I../../../../Middlewares/stm32-mw-camera/sensors/imx335 -I../../../../Middlewares/stm32-mw-camera/sensors/vd6g -I../../../../Middlewares/stm32-mw-camera/sensors/vd55g1 -I../../../../Middlewares/stm32-mw-camera/sensors/vd1943 -I../../../../Middlewares/stm32-mw-camera/sensors/ov5640 -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-stm32-2d-mw-2d-camera

clean-Middlewares-2f-stm32-2d-mw-2d-camera:
	-$(RM) ./Middlewares/stm32-mw-camera/cmw_camera.cyclo ./Middlewares/stm32-mw-camera/cmw_camera.d ./Middlewares/stm32-mw-camera/cmw_camera.o ./Middlewares/stm32-mw-camera/cmw_camera.su ./Middlewares/stm32-mw-camera/cmw_utils.cyclo ./Middlewares/stm32-mw-camera/cmw_utils.d ./Middlewares/stm32-mw-camera/cmw_utils.o ./Middlewares/stm32-mw-camera/cmw_utils.su

.PHONY: clean-Middlewares-2f-stm32-2d-mw-2d-camera

