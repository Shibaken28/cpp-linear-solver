import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

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
    (ご要望に基づき修正済み)
    """
    if not data:
        print("No data to plot.")
        return

    # --- フォント設定 ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'STIXGeneral', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # --- 横幅を狭く (figsize) ---
    fig, ax = plt.subplots(figsize=(7, 5))

    # データセットごとにプロット
    for rhs_count, residuals in data.items():
        # 横軸: 反復回数 (1, 2, 3, ...)
        iterations = range(1, len(residuals) + 1)
        # 折れ線グラフを描画
        ax.plot(iterations, residuals, linestyle='-', label=f'$L$=${rhs_count}$')

    # --- グラフの装飾 ---
    # Y軸を対数スケールに設定
    ax.set_yscale('log')

    # --- 軸の範囲 (修正) ---
    ax.set_xlim(left=0, right=1000)   # ★修正: 右端を1000に設定
    ax.set_ylim(bottom=1e-10)     # 一番下を10^-10に

    # --- ラベル (デカく、立体に) (修正) ---
    font_size_label = 16
    font_style = 'normal' # ★修正: 'italic' から 'normal' (立体) に変更
    ax.set_xlabel("Iteration Number", fontsize=font_size_label, style=font_style)
    ax.set_ylabel("Relative Residual Norm", fontsize=font_size_label, style=font_style)
    ax.set_title("", fontsize=font_size_label, style=font_style)

    # --- 凡例 (デカく、斜体に) (修正) ---
    legend_font_size = 14
    # ★修正: prop辞書で 'style': 'italic' を指定
    ax.legend(prop={'size': legend_font_size, 'style': 'italic'})
    ax.legend(fontsize=legend_font_size, loc='upper right')
    # --- ティック (もっとデカく、太字に) (修正) ---
    tick_font_size = 15 # ★修正: 14 -> 18 へ変更
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    # ティックの太字設定
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    
    # --- グリッドいらない ---
    # ax.grid(True, which="both", linestyle='--', linewidth=0.5) # コメントアウト

    # --- グラフ保存 ---
    plt.tight_layout()
    
    # --- Teamsの共有フォルダに置く ---
    # ★★★ 以下のパスを、ご自身のTeams共有フォルダのパスに変更してください ★★★
    # (Macの場合の例)
    # save_path = "/Users/YourUserName/Library/CloudStorage/OneDrive-YourOrganization/Teams Share - General/convergence_history.png"
    # (Windowsの場合の例)
    # save_path = "C:\\Users\\YourUserName\\OneDrive - YourOrganization\\Teams Share - General\\convergence_history.png"
    
    # とりあえずカレントディレクトリに保存
    save_path = "convergence_history.png"
    
    try:
        # 保存先ディレクトリが存在するか確認 (もしパスを指定した場合)
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.savefig(save_path, dpi=300)
        print(f"Plot successfully saved to: {os.path.abspath(save_path)}")
    except (IOError, FileNotFoundError, PermissionError) as e:
        print(f"Error saving file to '{save_path}': {e}")
        print("Please ensure the directory exists and you have write permissions.")


if __name__ == '__main__':

    data_filename = "out.txt"

    

    # 2. ファイルを読み込んでデータを解析

    convergence_data = parse_data(data_filename)

    

    # 3. データをプロット

    plot_data(convergence_data)