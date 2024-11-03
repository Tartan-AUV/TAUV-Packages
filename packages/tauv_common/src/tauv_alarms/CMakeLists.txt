add_custom_target(
    generate_alarms ALL
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/alarms.py ${CMAKE_CURRENT_SOURCE_DIR}/alarms.h
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/alarms.py ${CMAKE_CURRENT_SOURCE_DIR}/alarms.h
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_alarms.py ${CMAKE_CURRENT_SOURCE_DIR}/alarms.yaml --py ${CMAKE_CURRENT_SOURCE_DIR}/alarms.py --cpp ${CMAKE_CURRENT_SOURCE_DIR}/alarms.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/alarms.yaml ${CMAKE_CURRENT_SOURCE_DIR}/generate_alarms.py
    VERBATIM
)

set (CMAKE_CXX_STANDARD 17)
library(tauv_alarms alarm_client.cpp)
