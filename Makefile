# --- Compiler and General Flags ---
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -g

# --- Library Paths (for macOS with Homebrew) ---
EIGEN_PATH = $(shell brew --prefix eigen)
CXXFLAGS += -I$(EIGEN_PATH)/include/eigen3

# --- Project Files ---
TARGET = main_program
SRCS = main.cpp io_utils.cpp

# --- Build Rules ---
.PHONY: all clean clangd

all: $(TARGET)

$(TARGET): $(SRCS)
	@echo "Compiling and linking all sources..."
	$(CXX) $(CXXFLAGS) -o $@ $^

clangd:
	@echo "Generating compile_flags.txt for clangd..."
	@echo "$(CXXFLAGS)" | tr ' ' '\n' > compile_flags.txt
	@echo "Done."

clean:
	@echo "Cleaning up..."
	@# 実行ファイルと、OSが作る可能性のあるデバッグ用フォルダを削除
	rm -f $(TARGET)
	rm -rf $(TARGET).dSYM