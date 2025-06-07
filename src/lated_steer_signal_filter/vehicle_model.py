from dataclasses import dataclass
import numpy as np
from .runge_kutta import rk4_step
import numpy.typing as npt


@dataclass
class VehicleParameters:
    # 車両のパラメータ
    m: float
    I: float
    lf: float
    lr: float
    Kf: float
    Kr: float


@dataclass
class Result:
    time: npt.NDArray[np.floating]
    beta: npt.NDArray[np.floating]
    dot_phai: npt.NDArray[np.floating]


def sim_vehicle_model(param: VehicleParameters, delta, v, beta, dot_phai):
    m = param.m
    I = param.I
    lf = param.lf
    lr = param.lr
    Kf = param.Kf
    Kr = param.Kr

    a11 = -Kf / m - Kr / m
    a12 = -Kf * lf / m + Kr * lr / m
    a13 = Kf / m
    a21 = -Kf * lf / I + Kr * lr / I
    a22 = -Kf * lf**2 / I - Kr * lr**2 / I
    a23 = Kf * lf / I

    dot_beta = (a11 / v) * beta + (-1 + (a12 / v**2)) * dot_phai + (a13 / v) * delta
    ddot_phai = a21 * beta + (a22 / v) * dot_phai + a23 * delta

    return dot_beta, ddot_phai


def sim_vehicle(param: VehicleParameters, delta, v, t_finish, dt):
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
            param,
            variation[0],
            variation[1],
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
        beta, dot_phai = rk4_step(
            f, np.array([beta, dot_phai]), t, dt, np.array([input_delta, v])
        )

    # 結果を Pandas DataFrame に保存
    return Result(
        time=t_list,
        beta=np.array(beta_list),
        dot_phai=np.array(dot_phai_list),
    )
