import os, sys
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from numba import njit
import utils.jax_utils as jax_utils
from dynamics_models.dynamics_models_jax import *

CUDANUM = 0
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDANUM)

class InferEnv():
    def __init__(self, track, config, DT,
                 jrng=None, dyna_config=None) -> None:
        self.a_shape = 2
        self.track = track
        self.waypoints = track.waypoints
        self.diff = self.waypoints[1:, 1:3] - self.waypoints[:-1, 1:3]
        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        self.reference = None
        self.DT = DT
        self.config = config
        self.jrng = jax_utils.oneLineJaxRNG(0) if jrng is None else jrng
        self.state_frenet = jnp.zeros(6)
        self.norm_params = config.norm_params
        print('MPPI Model:', self.config.state_predictor)
        
        def RK4_fn(x0, u, Ddt, vehicle_dynamics_fn, args):
            # return x0 + vehicle_dynamics_fn(x0, u, *args) * Ddt # Euler integration
            # RK4 integration
            k1 = vehicle_dynamics_fn(x0, u, *args)
            k2 = vehicle_dynamics_fn(x0 + k1 * 0.5 * Ddt, u, *args)
            k3 = vehicle_dynamics_fn(x0 + k2 * 0.5 * Ddt, u, *args)
            k4 = vehicle_dynamics_fn(x0 + k3 * Ddt, u, *args)
            return x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * Ddt
            
        if self.config.state_predictor == 'dynamic_ST':
            @jax.jit
            def update_fn(x, u):
                x1 = x.copy()
                Ddt = 0.1
                def step_fn(i, x0):
                    args = (self.config.friction,)
                    return RK4_fn(x0, u, Ddt, vehicle_dynamics_st, args)
                x1 = jax.lax.fori_loop(0, int(self.DT/Ddt), step_fn, x1)
                return (x1, 0, x1-x)
            self.update_fn = update_fn
            
        elif self.config.state_predictor == 'kinematic_ST':
            @jax.jit
            def update_fn(x, u,):
                x_k = x.copy()[:5]
                Ddt = 0.1
                def step_fn(i, x0):
                    args = ()
                    return RK4_fn(x0, u, Ddt, vehicle_dynamics_ks, args)
                x_k = jax.lax.fori_loop(0, int(self.DT/Ddt), step_fn, x_k)
                x1 = x.at[:5].set(x_k)
                return (x1, 0, x1-x)
            self.update_fn = update_fn
            
    @partial(jax.jit, static_argnums=(0,))
    def step(self, x, u, rng_key=None, dyna_norm_param=None):
        return self.update_fn(x, u * self.norm_params[0, :2]/2)
    
    @partial(jax.jit, static_argnums=(0,))
    def reward_fn_sey(self, s, reference):
        """
        reward function for the state s with respect to the reference trajectory
        """
        sey_cost = -jnp.linalg.norm(reference[1:, 4:6] - s[:, :2], ord=1, axis=1)
        # vel_cost = -jnp.linalg.norm(reference[1:, 3] - s[:, 3])
        # yaw_cost = -jnp.abs(jnp.sin(reference[1:, 4]) - jnp.sin(s[:, 4])) - \
        #     jnp.abs(jnp.cos(reference[1:, 4]) - jnp.cos(s[:, 4]))
            
        return sey_cost
    
    def update_waypoints(self, waypoints):
        self.waypoints = waypoints
        self.diff = self.waypoints[1:, 1:3] - self.waypoints[:-1, 1:3]
        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
    
    @partial(jax.jit, static_argnums=(0,))
    def reward_fn_xy(self, state, reference):
        """
        reward function for the state s with respect to the reference trajectory
        """
        xy_cost = -jnp.linalg.norm(reference[1:, :2] - state[:, :2], ord=1, axis=1)
        vel_cost = -jnp.linalg.norm(reference[1:, 2] - state[:, 3])
        yaw_cost = -jnp.abs(jnp.sin(reference[1:, 3]) - jnp.sin(state[:, 4])) - \
            jnp.abs(jnp.cos(reference[1:, 4]) - jnp.cos(state[:, 4]))
            
        # return 20*xy_cost + 15*vel_cost + 1*yaw_cost
        return xy_cost
    
    
    def calc_ref_trajectory_kinematic(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        n_state = 4
        n_steps = 10
        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((n_state, n_steps + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 4.5] = np.abs(
            cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
        )
        cyaw[cyaw - state.yaw < -4.5] = np.abs(
            cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
        )
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj
    
    @partial(jax.jit, static_argnums=(0,3))
    def get_refernece_traj_jax(self, state, target_speed, n_steps=10):
        _, dist, _, _, ind = nearest_point_jax(jnp.array([state[0], state[1]]), 
                                           self.waypoints[:, (1, 2)], jnp.array(self.diff))
        
        speed = target_speed
        speeds = jnp.ones(n_steps) * speed
        
        reference = get_reference_trajectory_jax(speeds, dist, ind, 
                                            self.waypoints.copy(), int(n_steps),
                                            self.waypoints_distances.copy(), DT=self.DT)
        orientation = state[4]
        reference = reference.at[:, 3].set(
            jnp.where(reference[:, 3] - orientation > 5, 
                  reference[:, 3] - 2 * jnp.pi, 
                  reference[:, 3])
        )
        reference = reference.at[:, 3].set(
            jnp.where(reference[:, 3] - orientation < -5, 
                  reference[:, 3] + 2 * jnp.pi, 
                  reference[:, 3])
        )
        
        return reference, ind
    
    def get_refernece_traj(self, state, target_speed=None, n_steps=10, vind=5, speed_factor=1.0):
        _, dist, _, _, ind = nearest_point(np.array([state[0], state[1]]), 
                                           self.waypoints[:, (1, 2)].copy(), self.diff)
        
        if target_speed is None:
            # speed = self.waypoints[ind, vind] * speed_factor
            # speed = np.minimum(self.waypoints[ind, vind] * speed_factor, 20.)
            speed = state[3]
        else:
            speed = target_speed
        
        # if ind < self.waypoints.shape[0] - self.n_steps:
        #     speeds = self.waypoints[ind:ind+self.n_steps, vind]
        # else:
        speeds = np.ones(n_steps) * speed
        
        reference = get_reference_trajectory(speeds, dist, ind, 
                                            self.waypoints.copy(), int(n_steps),
                                            self.waypoints_distances.copy(), DT=self.DT)
        orientation = state[4]
        reference[3, :][reference[3, :] - orientation > 5] = np.abs(
            reference[3, :][reference[3, :] - orientation > 5] - (2 * np.pi))
        reference[3, :][reference[3, :] - orientation < -5] = np.abs(
            reference[3, :][reference[3, :] - orientation < -5] + (2 * np.pi))
        
        # reference[2] = np.where(reference[2] - speed > 5.0, speed + 5.0, reference[2])
        self.reference = reference.T
        return reference.T, ind

@jax.jit
def get_reference_trajectory_jax(predicted_speeds, dist_from_segment_start, idx, 
                             waypoints, n_steps, waypoints_distances, DT):
    total_length = jnp.sum(waypoints_distances)
    s_relative = jnp.concatenate([
        jnp.array([dist_from_segment_start]),
        predicted_speeds * DT
    ]).cumsum()
    s_relative = s_relative % total_length  
    rolled_distances = jnp.roll(waypoints_distances, -idx)
    wp_dist_cum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(rolled_distances)])
    index_relative = jnp.searchsorted(wp_dist_cum, s_relative, side='right') - 1
    index_relative = jnp.clip(index_relative, 0, len(rolled_distances) - 1)
    index_absolute = (idx + index_relative) % (waypoints.shape[0] - 1)
    next_index = (index_absolute + 1) % (waypoints.shape[0] - 1)
    seg_start = wp_dist_cum[index_relative]
    seg_len = rolled_distances[index_relative]
    t = (s_relative - seg_start) / seg_len
    p0 = waypoints[index_absolute][:, 1:3]
    p1 = waypoints[next_index][:, 1:3]
    interpolated_positions = p0 + (p1 - p0) * t[:, jnp.newaxis]
    s0 = waypoints[index_absolute][:, 0]
    s1 = waypoints[next_index][:, 0]
    interpolated_s = (s0 + (s1 - s0) * t) % waypoints[-1, 0]  
    yaw0 = waypoints[index_absolute][:, 3]
    yaw1 = waypoints[next_index][:, 3]
    interpolated_yaw = yaw0 + (yaw1 - yaw0) * t
    interpolated_yaw = (interpolated_yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
    v0 = waypoints[index_absolute][:, 5]
    v1 = waypoints[next_index][:, 5]
    interpolated_speed = v0 + (v1 - v0) * t
    reference = jnp.stack([
        interpolated_positions[:, 0],
        interpolated_positions[:, 1],
        interpolated_speed,
        interpolated_yaw,
        interpolated_s,
        jnp.zeros_like(interpolated_speed),
        jnp.zeros_like(interpolated_speed)
    ], axis=1)
    return reference

@jax.jit
def nearest_point_jax(point, trajectory, diffs):
    # diffs = trajectory[1:] - trajectory[:-1]                    
    l2s = jnp.sum(diffs**2, axis=1) + 1e-8                    
    dots = jnp.sum((point - trajectory[:-1]) * diffs, axis=1) 
    t = jnp.clip(dots / l2s, 0., 1.)   
    projections = trajectory[:-1] + diffs * t[:, None]
    dists = jnp.linalg.norm(point - projections, axis=1)      
    min_dist_segment = jnp.argmin(dists)                
    dist_from_segment_start = jnp.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])          
    return projections[min_dist_segment],dist_from_segment_start, dists[min_dist_segment], t[min_dist_segment], min_dist_segment
    



@njit(cache=True)
def nearest_point(point, trajectory, diffs):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    # diffs = trajectory[1:, :] - trajectory[:-1, :]
    # diffs = self.diff
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / (l2s + 1e-8)
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])
    return projections[min_dist_segment], dist_from_segment_start, dists[min_dist_segment], t[
        min_dist_segment], min_dist_segment


# @njit(cache=True)
def get_reference_trajectory(predicted_speeds, dist_from_segment_start, idx, 
                             waypoints, n_steps, waypoints_distances, DT):
    s_relative = np.zeros((n_steps + 1,))
    s_relative[0] = dist_from_segment_start
    s_relative[1:] = predicted_speeds * DT
    s_relative = np.cumsum(s_relative)

    waypoints_distances_relative = np.cumsum(np.roll(waypoints_distances, -idx))

    index_relative = np.int_(np.ones((n_steps + 1,)))
    for i in range(n_steps + 1):
        index_relative[i] = (waypoints_distances_relative <= s_relative[i]).sum()
    index_absolute = np.mod(idx + index_relative, waypoints.shape[0] - 1)

    segment_part = s_relative - (
            waypoints_distances_relative[index_relative] - waypoints_distances[index_absolute])

    t = (segment_part / waypoints_distances[index_absolute])
    # print(np.all(np.logical_and((t < 1.0), (t > 0.0))))

    position_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, (1, 2)] -
                        waypoints[index_absolute][:, (1, 2)])
    position_diff_s = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 0] -
                        waypoints[index_absolute][:, 0])
    orientation_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 3] -
                            waypoints[index_absolute][:, 3])
    speed_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 5] -
                    waypoints[index_absolute][:, 5])

    interpolated_positions = waypoints[index_absolute][:, (1, 2)] + (t * position_diffs.T).T
    interpolated_s = waypoints[index_absolute][:, 0] + (t * position_diff_s)
    interpolated_s[np.where(interpolated_s > waypoints[-1, 0])] -= waypoints[-1, 0]
    interpolated_orientations = waypoints[index_absolute][:, 3] + (t * orientation_diffs)
    interpolated_orientations = (interpolated_orientations + np.pi) % (2 * np.pi) - np.pi
    interpolated_speeds = waypoints[index_absolute][:, 5] + (t * speed_diffs)
    
    reference = np.array([
        # Sort reference trajectory so the order of reference match the order of the states
        interpolated_positions[:, 0],
        interpolated_positions[:, 1],
        interpolated_speeds,
        interpolated_orientations,
        # Fill zeros to the rest so number of references mathc number of states (x[k] - ref[k])
        interpolated_s,
        np.zeros(len(interpolated_speeds)),
        np.zeros(len(interpolated_speeds))
    ])
    return reference


