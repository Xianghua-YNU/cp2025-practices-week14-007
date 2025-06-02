import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

class TestVanDerPolOscillator(unittest.TestCase):
    def setUp(self):
        self.initial_state = np.array([1.0, 0.0])
        self.mu = 1.0
        self.omega = 1.0
        self.dt = 0.01
        
    def test_van_der_pol_ode(self):
        state = np.array([1.0, 2.0])
        t = 0.0
        derivative = van_der_pol_ode(t, state, self.mu, self.omega)
        
        # 验证导数的形状
        self.assertEqual(derivative.shape, (2,))
        
        # 验证导数的值
        expected_dx = 2.0  # v
        expected_dv = self.mu * (1 - 1.0**2) * 2.0 - self.omega**2 * 1.0  # mu(1-x^2)v - omega^2*x
        np.testing.assert_almost_equal(derivative[0], expected_dx, decimal=6)
        np.testing.assert_almost_equal(derivative[1], expected_dv, decimal=6)

    def test_solve_ode_basic_properties(self):
        # 调整t_span确保t_eval在范围内
        t_span = (0, 10)
        t, states = solve_ode(
            van_der_pol_ode,
            self.initial_state,
            t_span,
            self.dt,
            mu=self.mu,
            omega=self.omega
        )

        # 验证解的基本特性
        self.assertEqual(len(t), len(states))
        self.assertEqual(states.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(states)))
        self.assertTrue(t[0] >= t_span[0] and t[-1] <= t_span[1])

    def test_limit_cycle_analysis(self):
        """测试极限环分析"""
        # 生成测试数据：模拟极限环行为
        t = np.linspace(0, 50, 5000)
        x = 2 * np.cos(t)
        v = -2 * np.sin(t)
        states = np.column_stack((x, v))

        # 分析极限环
        amplitude, period = analyze_limit_cycle(states)

        # 验证结果
        self.assertGreater(amplitude, 0)
        self.assertGreater(period, 0)
        np.testing.assert_almost_equal(amplitude, 2.0, decimal=1)
        # 周期应该是2π/dt，因为t数组间隔是1
        np.testing.assert_almost_equal(period*self.dt, 2*np.pi, decimal=1)

def van_der_pol_ode(t, state, mu, omega):
    x, v = state
    dx = v
    dv = mu * (1 - x**2) * v - omega**2 * x
    return np.array([dx, dv])

def solve_ode(ode_func, initial_state, t_span, dt, mu, omega):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        ode_func, 
        t_span, 
        initial_state, 
        method='RK45', 
        t_eval=t_eval, 
        args=(mu, omega),
        rtol=1e-6, 
        atol=1e-6
    )
    return sol.t, sol.y.T

def analyze_limit_cycle(states):
    """
    分析极限环的振幅和周期
    
    参数:
    states: 状态变量数组，形状为 (n_points, 2)，第一列是x，第二列是v
    
    返回:
    amplitude: 振幅
    period: 周期（以时间步长为单位）
    """
    x = states[:, 0]
    
    # 寻找峰值
    peaks, _ = find_peaks(x)
    
    # 计算振幅
    if len(peaks) < 2:
        raise ValueError("无法找到足够的峰值来分析周期")
    
    # 计算振幅（假设对称）
    amplitude = np.mean(x[peaks])
    
    # 计算周期（平均峰值间隔）
    periods = np.diff(peaks)
    period = np.mean(periods)
    
    return amplitude, period

if __name__ == '__main__':
    unittest.main()    
