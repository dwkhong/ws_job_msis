#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}[INFO]${NC} Building SIMTOS Framework..."

CXX=g++
CXXFLAGS="-std=c++17 -O2 -Wall"
INCLUDES="-Iinclude"
SOURCES="main.cpp src/SocketConnection.cpp src/DeviceManager.cpp src/StateMachine.cpp"
OUTPUT="-o simtos_main"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    LDFLAGS="-lws2_32 -lpthread"
else
    LDFLAGS="-lpthread"
fi

echo -e "${BLUE}[INFO]${NC} $CXX $CXXFLAGS $INCLUDES $SOURCES $OUTPUT $LDFLAGS"

if $CXX $CXXFLAGS $INCLUDES $SOURCES $OUTPUT $LDFLAGS; then
    echo -e "${GREEN}[SUCCESS]${NC} Build completed!"
    chmod +x simtos_main 2>/dev/null || true
    ls -lh simtos_main*
else
    echo -e "${RED}[ERROR]${NC} Build failed!"
    exit 1
fi
