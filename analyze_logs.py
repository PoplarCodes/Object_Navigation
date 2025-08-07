import pandas as pd
import matplotlib.pyplot as plt


def plot_global_eps_mean_rew(log_path):
    try:
        df = pd.read_csv(log_path, delimiter=":", error_bad_lines=False)
        print(f"Loaded log file from: {log_path}")

        # 提取 Global eps mean rew 数据
        global_eps_mean_rew = df[df['Field'] == 'Global eps mean rew']

        # 绘制曲线
        plt.figure(figsize=(10, 6))
        plt.plot(global_eps_mean_rew['Step'], global_eps_mean_rew['Value'], label='Global eps mean rew')
        plt.xlabel('Step')
        plt.ylabel('Reward Value')
        plt.title('Global eps mean rew Trend')
        plt.legend()
        plt.grid()
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{log_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Printing first 10 lines of the log file:")
        try:
            with open(log_path, 'r') as f:
                for _ in range(10):
                    print(f.readline().strip())
        except Exception as e:
            print(f"Failed to read log file: {e}")


if __name__ == "__main__":
    log_path = "/home/robot/Object-Goal-Navigation/tmp/models/exp1/train.log"

    plot_global_eps_mean_rew(log_path)