################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include stm32-vision-models-postprocessing/subdir.mk
-include ll_aton/subdir.mk
-include Utilities/lcd/subdir.mk
-include Middlewares/stm32-mw-camera/sensors/drivers/subdir.mk
-include Middlewares/stm32-mw-camera/sensors/subdir.mk
-include Middlewares/stm32-mw-camera/subdir.mk
-include Middlewares/ISP_Library/subdir.mk
-include Drivers/STM32N6xx_HAL_Driver/Src/subdir.mk
-include Drivers/STM32N6xx_HAL_Driver/subdir.mk
-include Drivers/CMSIS/subdir.mk
-include Drivers/BSP/STM32N6570-DK/subdir.mk
-include Drivers/BSP/Components/subdir.mk
-include Application/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(S_DEPS)),)
-include $(S_DEPS)
endif
ifneq ($(strip $(S_UPPER_DEPS)),)
-include $(S_UPPER_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

OPTIONAL_TOOL_DEPS := \
$(wildcard ../makefile.defs) \
$(wildcard ../makefile.init) \
$(wildcard ../makefile.targets) \


BUILD_ARTIFACT_NAME := STM32N6570-DK_GettingStarted_FaceDetection
BUILD_ARTIFACT_EXTENSION := elf
BUILD_ARTIFACT_PREFIX :=
BUILD_ARTIFACT := $(BUILD_ARTIFACT_PREFIX)$(BUILD_ARTIFACT_NAME)$(if $(BUILD_ARTIFACT_EXTENSION),.$(BUILD_ARTIFACT_EXTENSION),)

# Add inputs and outputs from these tool invocations to the build variables 
EXECUTABLES += \
STM32N6570-DK_GettingStarted_FaceDetection.elf \

MAP_FILES += \
STM32N6570-DK_GettingStarted_FaceDetection.map \

SIZE_OUTPUT += \
default.size.stdout \

OBJDUMP_LIST += \
STM32N6570-DK_GettingStarted_FaceDetection.list \

OBJCOPY_BIN += \
STM32N6570-DK_GettingStarted_FaceDetection.bin \


# All Target
all: main-build

# Main-build Target
main-build: STM32N6570-DK_GettingStarted_FaceDetection.elf secondary-outputs

# Tool invocations
STM32N6570-DK_GettingStarted_FaceDetection.elf STM32N6570-DK_GettingStarted_FaceDetection.map: $(OBJS) $(USER_OBJS) C:\FD\stm32ai-modelzoo-services\application_code\face_detection\STM32N6\Application\STM32N6570-DK\STM32CubeIDE\STM32N657xx.ld makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-gcc -o "STM32N6570-DK_GettingStarted_FaceDetection.elf" @"objects.list" $(USER_OBJS) $(LIBS) -mcpu=cortex-m55 -T"C:\FD\stm32ai-modelzoo-services\application_code\face_detection\STM32N6\Application\STM32N6570-DK\STM32CubeIDE\STM32N657xx.ld" --specs=nosys.specs -Wl,-Map="STM32N6570-DK_GettingStarted_FaceDetection.map" -Wl,--gc-sections -static -Wl,--cmse-implib -Wl,--out-implib=./secure_nsclib.o -L../../../../Middlewares/stedgeai-lib/Lib/GCC/ARMCortexM55 -Wl,--no-warn-rwx-segments --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -u _printf_float -Wl,--start-group -lc -lm -Wl,--end-group
	@echo 'Finished building target: $@'
	@echo ' '

default.size.stdout: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-size  $(EXECUTABLES)
	@echo 'Finished building: $@'
	@echo ' '

STM32N6570-DK_GettingStarted_FaceDetection.list: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-objdump -h -S $(EXECUTABLES) > "STM32N6570-DK_GettingStarted_FaceDetection.list"
	@echo 'Finished building: $@'
	@echo ' '

STM32N6570-DK_GettingStarted_FaceDetection.bin: $(EXECUTABLES) makefile objects.list $(OPTIONAL_TOOL_DEPS)
	arm-none-eabi-objcopy  -O binary $(EXECUTABLES) "STM32N6570-DK_GettingStarted_FaceDetection.bin"
	@echo 'Finished building: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) STM32N6570-DK_GettingStarted_FaceDetection.bin STM32N6570-DK_GettingStarted_FaceDetection.elf STM32N6570-DK_GettingStarted_FaceDetection.list STM32N6570-DK_GettingStarted_FaceDetection.map default.size.stdout
	-@echo ' '

secondary-outputs: $(SIZE_OUTPUT) $(OBJDUMP_LIST) $(OBJCOPY_BIN)

fail-specified-linker-script-missing:
	@echo 'Error: Cannot find the specified linker script. Check the linker settings in the build configuration.'
	@exit 2

warn-no-linker-script-specified:
	@echo 'Warning: No linker script specified. Check the linker settings in the build configuration.'

.PHONY: all clean dependents main-build fail-specified-linker-script-missing warn-no-linker-script-specified

-include ../makefile.targets
