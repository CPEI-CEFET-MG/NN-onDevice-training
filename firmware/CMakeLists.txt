# Versão mínima do CMake
cmake_minimum_required(VERSION 3.5)

# A linha "set(EXTRA_COMPONENT_DIRS static_images)" foi removida.
# O bloco de lógica do BSP foi removido.

# Esta é a linha mais importante: ela carrega o ESP-IDF
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Define o nome do seu projeto
project(firmware)