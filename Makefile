# ==========================================
# 設定
# ==========================================
CXX      = g++

# --- ライブラリ設定 (Eigen) ---
# HomebrewでインストールされたEigenのパスを取得
EIGEN_PATH = $(shell brew --prefix eigen)

# インクルードパスの設定
# 1. 自作コードのディレクトリ (src/common, src/algo)
# 2. Eigenのディレクトリ (brewのパス/include/eigen3)
INCLUDES = -Isrc -I$(EIGEN_PATH)/include/eigen3

CXXFLAGS = -Wall -O2 -std=c++17 $(INCLUDES)

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# ==========================================
# 自動検出ロジック
# ==========================================

# 1. ロジック（共通処理など）の全ファイルを検出
#    common, algo, algo_xxx などのディレクトリ内の.cppを全て拾います
LOGIC_SRCS = $(wildcard $(SRC_DIR)/common/*.cpp) \
             $(wildcard $(SRC_DIR)/algo/*.cpp) \
             $(wildcard $(SRC_DIR)/algo_*/*.cpp)

LOGIC_OBJS = $(LOGIC_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# 2. メイン関数のファイルを検出
MAIN_SRCS = $(wildcard $(SRC_DIR)/main/*.cpp)

# 3. 生成すべき実行ファイルのパスを自動計算
#    src/main/xxx.cpp -> bin/xxx
TARGETS = $(patsubst $(SRC_DIR)/main/%.cpp, $(BIN_DIR)/%, $(MAIN_SRCS))

# 4. エイリアス用（make io_sample で呼べるようにするため）
PROGRAM_NAMES = $(notdir $(TARGETS))

# ==========================================
# ビルドターゲット
# ==========================================

all: $(TARGETS)

# コマンド名短縮用 (make io_sample)
$(PROGRAM_NAMES): %: $(BIN_DIR)/%

.PHONY: all clean $(PROGRAM_NAMES)

# ==========================================
# ルール定義
# ==========================================

# ★汎用リンクルール★
# メインの.o と ロジック全部入り.o をリンクして実行ファイルを作る
$(BIN_DIR)/%: $(OBJ_DIR)/main/%.o $(LOGIC_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# .cpp -> .o のコンパイルルール
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# クリーンアップ
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)