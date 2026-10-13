import math
import unittest
from types import SimpleNamespace
import torch
from window_reward import WindowReward, install, local_delta, yaw_xyzw


class WindowTests(unittest.TestCase):
    def state(self, dt=.02, count=1):
        s = WindowReward(count, 'cpu', dt)
        p, y = torch.zeros(count, 2), torch.zeros(count)
        s.reset(torch.arange(count), p, y, p, y)
        return s, p, y

    def test_perfect_progress_and_turn(self):
        s, p, y = self.state()
        for i in range(1, 126):
            p = torch.tensor([[i*.01, math.sin(i*.02)*.1]])
            y = torch.tensor([i*.03])
            s.advance(p, y, p, y)
            self.assertEqual(float(s.position_penalty+s.yaw_penalty), 0.)

    def test_stopped_robot_target_moves_and_boundary_is_scored(self):
        s, p, y = self.state()
        for i in range(1, 51):
            s.advance(p, y, torch.tensor([[i*.02, 0.]]), y)
        self.assertAlmostEqual(float(s.position_penalty), 3.5, places=5)
        self.assertEqual(int(s.age), 0)
        self.assertEqual(float(s.features[0, :4].abs().sum()), 0.)
        s.advance(p, y, torch.tensor([[1.02, 0.]]), y)
        self.assertAlmostEqual(float(s.distance_error), .02, places=5)

    def test_independent_global_offset_and_heading_invariance(self):
        s, p, y = self.state()
        robot = torch.tensor([[10., 20.]])
        ref = torch.tensor([[-7., -8.]])
        ry, ty = torch.tensor([math.pi/2]), torch.tensor([0.])
        s.reset(torch.tensor([0]), robot, ry, ref, ty)
        s.advance(robot+torch.tensor([[0., .5]]), ry, ref+torch.tensor([[.5, 0.]]), ty)
        self.assertLess(float(s.position_penalty), 1e-10)

    def test_yaw_wrap_crossing_pi(self):
        s, p, y = self.state()
        s.reset(torch.tensor([0]), p, torch.tensor([math.pi-.1]), p, y)
        s.advance(p, torch.tensor([-math.pi+.1]), p, torch.tensor([.2]))
        self.assertLess(float(s.yaw_penalty), 1e-10)

    def test_partial_reset_does_not_touch_other_env(self):
        s, p, y = self.state(count=2)
        s.advance(p, y, p+1, y)
        other = s.features[1].clone()
        s.reset(torch.tensor([0]), p, y, p+1, y)
        self.assertTrue(torch.equal(other, s.features[1]))
        self.assertEqual(s.age.tolist(), [0, 1])
        s.advance(p, y, p+1, y)
        self.assertEqual(float(s.position_penalty[0]), 0.)

    def test_frequency_integral_converges(self):
        totals = []
        for dt in (.02, .01):
            s, p, y = self.state(dt)
            total = 0.
            for i in range(1, round(1/dt)+1):
                s.advance(p, y, torch.tensor([[i*dt, 0.]]), y)
                total += float(s.position_penalty)*dt
            totals.append(total)
        self.assertLess(abs(totals[0]-totals[1]), .02)

    def test_gradients_not_saturated_far_from_target(self):
        s, p, y = self.state()
        target = torch.tensor([[2., 0.]], requires_grad=True)
        s.advance(p, y, target, y)
        s.position_penalty.sum().backward()
        self.assertEqual(float(target.grad[0, 0]), 4.)

    def test_quaternion_convention(self):
        q = torch.tensor([[0., 0., math.sin(.6), math.cos(.6)]])
        self.assertAlmostEqual(float(yaw_xyzw(q)), 1.2, places=6)
        torch.testing.assert_close(yaw_xyzw(q*1.073), yaw_xyzw(q))

    def test_install_reward_sign_dt_and_reset(self):
        e = SimpleNamespace(num_envs=2, device='cpu', dt=.02,
            root_states=torch.zeros(2, 13), _ref_root_pos=torch.zeros(2, 3),
            _ref_root_rot=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1),
            reward_names=[], reward_functions=[], reward_scales={}, episode_sums={})
        e.root_states[:, 6] = 1.
        def reward():
            e.r = sum(f()*e.reward_scales[n] for n,f in zip(e.reward_names,e.reward_functions))
        e.compute_reward = reward
        e.reset_idx = lambda ids: None
        s = install(e)
        e._ref_root_pos[:, 0] = .25
        e.compute_reward()
        torch.testing.assert_close(e.r, torch.full((2,), -.0025))
        e.reset_idx(torch.tensor([0]))
        e.compute_reward()
        self.assertEqual(float(e.r[0]), 0.)
        self.assertLess(float(e.r[1]), 0.)


if __name__ == '__main__':
    unittest.main()
