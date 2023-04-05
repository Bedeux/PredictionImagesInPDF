# Install script for directory: /home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib

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
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-glib.so.8.21.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-glib.so.8"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu" TYPE SHARED_LIBRARY FILES
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/glib/libpoppler-glib.so.8.21.0"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/glib/libpoppler-glib.so.8"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-glib.so.8.21.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-glib.so.8"
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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-glib.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-glib.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-glib.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu" TYPE SHARED_LIBRARY FILES "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/glib/libpoppler-glib.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-glib.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-glib.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-glib.so"
         OLD_RPATH "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/x86_64-linux-gnu/libpoppler-glib.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/poppler/glib" TYPE FILE FILES
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler-action.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler-date.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler-document.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler-page.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler-attachment.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler-form-field.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler-annot.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler-layer.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler-movie.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler-media.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/glib/poppler-structure-element.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/glib/poppler-enums.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/glib/poppler-features.h"
    "/home/bbordenave/personalProjects/PredictionImagesInPDF/poppler-21.09.0/build/glib/poppler-macros.h"
    )
endif()

