import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, List

def harmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    简谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    # TODO: 实现简谐振子的微分方程组
    # dx/dt = v
    # dv/dt = -omega^2 * x
    dx_dt = v
    dv_dt = -(omega**2)*x
    return np.array(dx_dt,dv_dt)
    

def anharmonic_oscillator_ode(state: np.ndarray, t: float, omega: float = 1.0) -> np.ndarray:
    """
    非谐振子的一阶微分方程组。
    
    参数:
        state: np.ndarray, 形状为(2,)的数组，包含位置x和速度v
        t: float, 当前时间（在这个系统中实际上没有使用）
        omega: float, 角频率
    
    返回:
        np.ndarray: 形状为(2,)的数组，包含dx/dt和dv/dt
    """
    x, v = state
    # TODO: 实现非谐振子的微分方程组
    # dx/dt = v
    # dv/dt = -omega^2 * x^3
    dx_dt = v
    dv_dt = -(omega**2)*(x**3)
    return np.array([dx_dt,dv_dt])

def rk4_step(ode_func: Callable, state: np.ndarray, t: float, dt: float, **kwargs) -> np.ndarray:
    """
    使用四阶龙格-库塔方法进行一步数值积分。
    
    参数:
        ode_func: Callable, 微分方程函数
        state: np.ndarray, 当前状态
        t: float, 当前时间
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        np.ndarray: 下一步的状态
    """
    # TODO: 实现RK4方法
    k1 = ode_func(state, t, **kwargs)
    k2 = ode_func(state + 0.5*dt*k1, t + 0.5*dt, **kwargs)
    k3 = ode_func(state + 0.5*dt*k2, t + 0.5*dt, **kwargs)
    k4 = ode_func(state + dt*k3, t + dt, **kwargs)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    

def solve_ode(ode_func: Callable, initial_state: np.ndarray, t_span: Tuple[float, float], 
              dt: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解常微分方程组。
    
    参数:
        ode_func: Callable, 微分方程函数
        initial_state: np.ndarray, 初始状态
        t_span: Tuple[float, float], 时间范围 (t_start, t_end)
        dt: float, 时间步长
        **kwargs: 传递给ode_func的额外参数
    
    返回:
        Tuple[np.ndarray, np.ndarray]: (时间点数组, 状态数组)
    """
    # TODO: 实现ODE求解器
    t_start,t_end = t_span
    t = np.arange(t_start,t_end + dt, dt)
    states = np.zeros((len(t), len(initial_state)))
    states[0] = initial_state
    for i in range(1, len(t)):
        states[i] = rk4_step(ode_func,states[i-1],t[i-1],dt,**kwargs)
    return t, states

def plot_time_evolution(t: np.ndarray, states: np.ndarray, title: str) -> None:
    """
    绘制状态随时间的演化。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现时间演化图的绘制
    plt.figure(figsize=(12, 6))
    # 绘制位置随时间变化
    plt.subplot(2, 1, 1)
    plt.plot(t, states[:, 0], 'b-', label='位置 x')
    plt.xlabel('时间')
    plt.ylabel('位置')
    plt.title(f'{title} - 位置随时间变化')
    plt.grid(True)
    plt.legend()
    # 绘制速度随时间变化
    plt.subplot(2, 1, 2)
    plt.plot(t, states[:, 1], 'r-', label='速度 v')
    plt.xlabel('时间')
    plt.ylabel('速度')
    plt.title(f'{title} - 速度随时间变化')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_phase_space(states: np.ndarray, title: str) -> None:
    """
    绘制相空间轨迹。
    
    参数:
        states: np.ndarray, 状态数组
        title: str, 图标题
    """
    # TODO: 实现相空间图的绘制
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 1], 'g-', linewidth=1.5)
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.title(f'{title} - 相空间轨迹')
    plt.grid(True)
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(0, color='k', linestyle='--', alpha=0.3)
    plt.show()

def analyze_period(t: np.ndarray, states: np.ndarray) -> float:
    """
    分析振动周期。
    
    参数:
        t: np.ndarray, 时间点数组
        states: np.ndarray, 状态数组
    
    返回:
        float: 估计的振动周期
    """
    # TODO: 实现周期分析
    # 获取位置数据
    x = states[:, 0]
    # 找到所有过零点（从正变负的穿越点）
    zero_crossings = np.where(np.diff(np.sign(x)) < 0)[0]
    # 需要至少两个过零点才能计算周期
    if len(zero_crossings) < 2:
        print("警告：数据不足，无法计算周期")
        return 0.0
    # 计算连续过零点之间的时间差
    periods = np.diff(t[zero_crossings])
    # 计算平均周期（跳过前两个瞬态周期）
    if len(periods) > 2:
        avg_period = np.mean(periods[1:]) * 2  # 乘以2因为相邻过零点间隔是半个周期
    else:
        avg_period = np.mean(periods) * 2
    return avg_period

def main():
    # 设置参数
    omega = 1.0
    t_span = (0, 50)
    dt = 0.01
    
    # TODO: 任务1 - 简谐振子的数值求解
    # 1. 设置初始条件 x(0)=1, v(0)=0
    # 2. 求解方程
    # 3. 绘制时间演化图
    print("任务1: 简谐振子分析")
    initial_state = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
    t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
    # 绘制时间演化图
    plot_time_evolution(t, states, "简谐振子")
    # 绘制相空间轨迹
    plot_phase_space(states, "简谐振子")
    # 分析周期
    period = analyze_period(t, states)
    print(f"简谐振子估计周期: {period:.4f} (理论值: {2*np.pi/omega:.4f})")
    
    # TODO: 任务2 - 振幅对周期的影响分析
    # 1. 使用不同的初始振幅
    # 2. 分析周期变化
    print("\n任务2: 振幅对周期的影响分析")
    amplitudes = [0.5, 1.0, 1.5, 2.0]
    harmonic_periods = []
    for amp in amplitudes:
        initial_state = np.array([amp, 0.0])
        t, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        period = analyze_period(t, states)
        harmonic_periods.append(period)
        print(f"振幅 {amp}: 周期 = {period:.4f}")
    # 绘制振幅-周期关系图
    plt.figure()
    plt.plot(amplitudes, harmonic_periods, 'bo-')
    plt.axhline(2*np.pi/omega, color='r', linestyle='--', label='理论周期')
    plt.xlabel('初始振幅')
    plt.ylabel('周期')
    plt.title('简谐振子: 振幅对周期的影响')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # TODO: 任务3 - 非谐振子的数值分析
    # 1. 求解非谐振子方程
    # 2. 分析不同振幅的影响
    print("\n任务3: 非谐振子分析")
    anharmonic_periods = []
    
    for amp in amplitudes:
        initial_state = np.array([amp, 0.0])
        t, states = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        # 绘制时间演化图（只绘制最后一个振幅）
        if amp == amplitudes[-1]:
            plot_time_evolution(t, states, "非谐振子")
        # 分析周期
        period = analyze_period(t, states)
        anharmonic_periods.append(period)
        print(f"振幅 {amp}: 周期 = {period:.4f}")
        
    # TODO: 任务4 - 相空间分析
    # 1. 绘制相空间轨迹
    # 2. 比较简谐和非谐振子
    print("\n任务4: 相空间分析")
    # 比较不同振幅下的相空间轨迹
    plt.figure(figsize=(10, 8))
    # 绘制简谐振子相空间
    for i, amp in enumerate(amplitudes):
        initial_state = np.array([amp, 0.0])
        _, states = solve_ode(harmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        plt.plot(states[:, 0], states[:, 1], label=f'简谐 (A={amp})')
    plt.title('简谐振子相空间轨迹')
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.grid(True)
    plt.legend()
    plt.show()
    # 绘制非谐振子相空间
    plt.figure(figsize=(10, 8))
    for i, amp in enumerate(amplitudes):
        initial_state = np.array([amp, 0.0])
        _, states = solve_ode(anharmonic_oscillator_ode, initial_state, t_span, dt, omega=omega)
        plt.plot(states[:, 0], states[:, 1], label=f'非谐 (A={amp})')
    plt.title('非谐振子相空间轨迹')
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.grid(True)
    plt.legend()
    plt.show()
    # 比较两种振子的振幅-周期关系
    plt.figure()
    plt.plot(amplitudes, harmonic_periods, 'bo-', label='简谐振子')
    plt.plot(amplitudes, anharmonic_periods, 'ro-', label='非谐振子')
    plt.xlabel('初始振幅')
    plt.ylabel('周期')
    plt.title('振幅对周期的影响比较')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
