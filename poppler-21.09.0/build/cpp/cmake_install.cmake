# Install script for directory: /home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-cpp.so.0.9.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-cpp.so.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu" TYPE SHARED_LIBRARY FILES
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/cpp/libpoppler-cpp.so.0.9.0"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/cpp/libpoppler-cpp.so.0"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-cpp.so.0.9.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-cpp.so.0"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build:"
           NEW_RPATH "")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-cpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-cpp.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu" TYPE SHARED_LIBRARY FILES "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/cpp/libpoppler-cpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-cpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-cpp.so"
         OLD_RPATH "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-cpp.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/poppler/cpp" TYPE FILE FILES
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-destination.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-document.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-embedded-file.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-font.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-font-private.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-global.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-image.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-page.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-page-renderer.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-page-transition.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-rectangle.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/cpp/poppler-toc.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/cpp/poppler_cpp_export.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/cpp/poppler-version.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/cpp/tests/cmake_install.cmake")

endif()

