cmake_minimum_required(VERSION 3.18)

project(lazyVec)

OPTION(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE "Enable the serial CPU accelerator" ON)

find_package(vikunja REQUIRED)

alpaka_add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE include/)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
  vikunja::vikunja
)
