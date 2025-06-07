import numpy as np
import pandas as pd
from scipy.integrate import odeint
from runge_kutta import rk4_step
import matplotlib.pyplot as plt


def sim_vehicle(delta, v, t_finish, dt):

    def sim_vehicle_model(delta, v, time, beta, dot_phai):
        m = 2000
        I = 3000
        lf = 1.51
        lr = 1.49
        Kf = 10**5
        Kr = 10**5

        a11 = -Kf / m - Kr / m
        a12 = -Kf * lf / m + Kr * lr / m
        a13 = Kf / m
        a21 = -Kf * lf / I + Kr * lr / I
        a22 = -Kf * lf**2 / I - Kr * lr**2 / I
        a23 = Kf * lf / I

        dot_beta = (a11 / v) * beta + (-1 + (a12 / v**2)) * dot_phai + (a13 / v) * delta
        ddot_phai = a21 * beta + (a22 / v) * dot_phai + a23 * delta

        return dot_beta, ddot_phai

    # #########
    # v = 10 # m/s
    # t_finish = 5
    # dt = 0.001  # シミュレーションの時間ステップ

    # 初期条件
    beta = 0.0
    dot_phai = 0.0

    # 時間点のリストを作成
    steps = int(t_finish / dt)
    t_list = np.linspace(0, t_finish, steps)

    # 結果を保存するリスト
    beta_list = []
    dot_phai_list = []

    def f(current_integrate_value, t, variation):
        db, ddp = sim_vehicle_model(
            variation[0],
            variation[1],
            t,
            current_integrate_value[0],
            current_integrate_value[1],
        )
        return [db, ddp]

    # シミュレーションループ
    for i, t in enumerate(t_list):
        # 操舵入力の例
        input_delta = delta.iloc[i - 1]

        # 次のステップのために更新する前に現在の値を保存
        beta_list.append(beta)
        dot_phai_list.append(dot_phai)

        # RK4 を使って次のステップを計算
        beta, dot_phai = rk4_step(f, [beta, dot_phai], t, dt, [input_delta, v])

    # 結果を Pandas DataFrame に保存
    df = pd.DataFrame({"time": t_list, "beta": beta_list, "dot_phai": dot_phai_list})

    # 結果の最初の数行を表示
    print(df.head())

    # 必要に応じて結果を CSV ファイルに保存することもできます
    # df.to_csv('vehicle_sim_result.csv', index=False)
    # ans = sim_vehicle_model(1,1,1,1,1)
    # print(ans)

    # Integrate dot_phai to get phai (yaw angle)
    df["phai"] = df["dot_phai"].cumsum() * dt

    # Initial position
    x = 0.0
    y = 0.0

    # Lists to store position data
    x_list = [x]
    y_list = [y]
    current_phai_list = [x]
    current_beta_list = [x]
    # Calculate x and y coordinates
    for i in range(1, len(df)):
        current_phai = df["phai"].iloc[i - 1]
        current_beta = df["beta"].iloc[i - 1]

        current_phai_list.append(current_phai)
        current_beta_list.append(current_beta)

        # Kinematic equations for vehicle position
        # Assuming constant velocity 'v' for simplicity, as it was given as a constant 'v=17'
        dot_x = v * np.cos(current_phai + current_beta)
        dot_y = v * np.sin(current_phai + current_beta)

        x += dot_x * dt
        y += dot_y * dt

        x_list.append(x)
        y_list.append(y)

    df["x"] = x_list
    df["y"] = y_list
    df["current_phai"] = current_phai_list
    df["current_beta"] = current_beta_list
    # Plotting the trajectory

    plt.figure(figsize=(10, 8))
    plt.plot(df["x"], df["y"], label="車両の軌道")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("車両の軌道シミュレーション")
    plt.grid(True)
    plt.axis("equal")  # Equal scaling for x and y axes
    plt.legend()
    plt.show()

    return df


velocity = 16  # m/s
finish_time = 5  # s
time_step = 0.001  # s
delta = pd.Series(np.zeros(int(finish_time / time_step)))  # 操舵入力の初期化
delta.iloc[0 : int(0.5 / time_step)] = 1.0  # 最初の0.5秒間は操舵入力を1.0に設定

result = sim_vehicle(delta, velocity, finish_time, time_step)
