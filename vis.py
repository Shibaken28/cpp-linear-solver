import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def parse_data(filename):
    """
    指定された形式のデータファイルを読み込み、
    プロット用のデータ構造(辞書)に変換します。
    """
    data = {}
    try:
        with open(filename, 'r') as f:
            # 1行目: テストケースの総数 N
            num_test_cases = int(f.readline().strip())

            for _ in range(num_test_cases):
                # L_i: 右辺ベクトル数
                rhs_count = int(f.readline().strip())
                # M: 反復回数
                iterations = int(f.readline().strip())
                
                residuals = []
                for _ in range(iterations):
                    # 残差データを読み込む
                    residuals.append(float(f.readline().strip()))
                
                # 辞書に格納 (キー: ベクトル数, 値: 残差リスト)
                data[rhs_count] = residuals
                
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except (ValueError, IndexError):
        print(f"Error: The file '{filename}' has an invalid format.")
        return None
        
    return data

def plot_data(data):
    """
    解析されたデータを受け取り、折れ線グラフを描画します。
    """
    if not data:
        print("No data to plot.")
        return

    plt.rcParams['font.family'] = 'serif'
    # 使用するセリフ体フォントの優先順位を指定
    plt.rcParams['font.serif'] = ['Times New Roman', 'STIXGeneral', 'serif']
    # 数式（軸の指数表記など）のフォントもTimes系に合うものに設定
    plt.rcParams['mathtext.fontset'] = 'stix'
    # グラフの作成
    fig, ax = plt.subplots(figsize=(10, 6))

    # データセットごとにプロット
    for rhs_count, residuals in data.items():
        # 横軸: 反復回数 (1, 2, 3, ...)
        iterations = range(1, len(residuals) + 1)
        # 折れ線グラフを描画
        ax.plot(iterations, residuals, linestyle='-', label=f'L={rhs_count}')

    # --- グラフの装飾 ---
    # Y軸を対数スケールに設定
    ax.set_yscale('log')

    # ラベルとタイトル
    ax.set_xlabel("Iteration Count", fontsize=12)
    ax.set_ylabel("Relative Residual", fontsize=12)
    ax.set_title("Convergence History of Iterative Solver", fontsize=14)

    # 凡例を表示
    ax.legend()
    
    # グリッドを表示
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)

    # グラフ保存
    plt.tight_layout()
    plt.savefig("convergence_history.png", dpi=300)


if __name__ == '__main__':
    data_filename = "out.txt"
    
    # 2. ファイルを読み込んでデータを解析
    convergence_data = parse_data(data_filename)
    
    # 3. データをプロット
    plot_data(convergence_data)