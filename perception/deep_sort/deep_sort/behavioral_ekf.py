import numpy as np
import scipy.linalg

# Re-export for gating (linear_assignment can use kalman_filter.chi2inv95 or this)
from . import kalman_filter as _kf
chi2inv95 = _kf.chi2inv95


def _vphi_from_xy(vx, vy):
    v = np.sqrt(vx * vx + vy * vy)
    phi = np.arctan2(vy, vx) if v > 1e-8 else 0.0
    return v, phi


def _xy_from_vphi(v, phi):
    return v * np.cos(phi), v * np.sin(phi)


class BehavioralEKF:
    """
    Extended Kalman Filter with Coordinated Turn (CT) model 
    and Social Force behavior prediction.
    """
    def __init__(self, dt=0.05):
        self.dt = dt
        # State vector: [p_x, p_y, v, phi, omega]
        self.ndim = 5 
        
        # Tuning parameters for Social Force
        self.A_soc = 2.0  # Interaction strength
        self.B_soc = 0.5  # Interaction range
        self.pedestrian_radius = 0.3 # meters (or pixel equivalent)

        # Process Noise Covariance (Q) 
        # self._Q = np.diag([0.1, 0.1, 0.5, 0.1, 0.1]) ** 2
        self._Q = np.diag([1.0, 1.0, 1.9, 3.0, 3.0]) ** 2

        # Measurement Noise Covariance (R) - Assuming we measure [p_x, p_y]
        self._R = np.diag([0.2, 0.2]) ** 2

    def _compute_social_force(self, current_state, other_states):
        """Calculates the repulsive social force from neighboring pedestrians."""
        f_soc = np.array([0.0, 0.0])
        px, py = current_state[0], current_state[1]

        for neighbor in other_states:
            nx, ny = neighbor[0], neighbor[1]
            dist = np.hypot(nx - px, ny - py)
            
            if dist < 0.01: # Avoid division by zero for self or identical tracks
                continue
                
            if dist < 3.0: # Only care about neighbors within 3 meters/units
                # Normalized direction vector pointing AWAY from neighbor
                n_ij = np.array([px - nx, py - ny]) / dist
                # Exponential repulsion
                force_mag = self.A_soc * np.exp(((self.pedestrian_radius * 2) - dist) / self.B_soc)
                f_soc += force_mag * n_ij
                
        return f_soc

    def predict(self, mean, covariance, other_track_means=None):
        """
        EKF Prediction step using CT model + Social Force.
        """
        px, py, v, phi, omega = mean
        dt = self.dt

        # 1. Compute Social Force
        f_soc = np.array([0.0, 0.0])
        if other_track_means is not None and len(other_track_means) > 0:
            f_soc = self._compute_social_force(mean, other_track_means)

        # 2. State Transition (CT Model)
        x_pred = np.copy(mean)
        
        # Handle singularity when driving straight (omega approx 0)
        if abs(omega) < 1e-4:
            x_pred[0] += v * np.cos(phi) * dt
            x_pred[1] += v * np.sin(phi) * dt
        else:
            x_pred[0] += (v / omega) * (np.sin(phi + omega * dt) - np.sin(phi))
            x_pred[1] += (v / omega) * (np.cos(phi) - np.cos(phi + omega * dt))
            
        x_pred[3] += omega * dt

        # Apply Social Force as control input (F = ma, assume m=1)
        # We inject the force into the position prediction directly (0.5 * a * t^2)
        # x_pred[0] += 0.5 * f_soc[0] * (dt ** 2)
        # x_pred[1] += 0.5 * f_soc[1] * (dt ** 2)

        # 3. Calculate Jacobian (F) for Covariance update
        # Simplified Jacobian for the CT model
        F = np.eye(self.ndim)
        if abs(omega) < 1e-4:
            F[0, 2] = np.cos(phi) * dt
            F[0, 3] = -v * np.sin(phi) * dt
            F[1, 2] = np.sin(phi) * dt
            F[1, 3] = v * np.cos(phi) * dt
        else:
            F[0, 2] = (np.sin(phi + omega * dt) - np.sin(phi)) / omega
            F[0, 3] = (v / omega) * (np.cos(phi + omega * dt) - np.cos(phi))
            F[1, 2] = (np.cos(phi) - np.cos(phi + omega * dt)) / omega
            F[1, 3] = (v / omega) * (np.sin(phi + omega * dt) + np.sin(phi))

        # 4. Update Covariance
        covariance_pred = np.dot(F, np.dot(covariance, F.T)) + self._Q

        return x_pred, covariance_pred

    def update(self, mean, covariance, measurement):
        """
        EKF Update step. Measurement z = [p_x, p_y]
        """
        # Measurement matrix H (we only observe x and y)
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])

        # Innovation (Residual)
        z_pred = np.dot(H, mean)
        y = measurement - z_pred

        # Innovation Covariance (S)
        S = np.dot(H, np.dot(covariance, H.T)) + self._R

        # Kalman Gain (K)
        K = np.dot(covariance, np.dot(H.T, np.linalg.inv(S)))

        # Update State and Covariance
        new_mean = mean + np.dot(K, y)
        I = np.eye(self.ndim)
        new_covariance = np.dot((I - np.dot(K, H)), covariance)

        return new_mean, new_covariance


class BehavioralEKFFilter(object):
    """
    Drop-in replacement for KalmanFilter. Uses BehavioralEKF for (x, y) motion
    (CT model + social force) and constant velocity for (a, h).
    State is 8D: (x, y, a, h, vx, vy, va, vh); measurement is (x, y, a, h).
    """

    def __init__(self, dt=1.0):
        self._ekf = BehavioralEKF(dt=dt)
        self._dt = dt
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement (x, y, a, h). Returns 8D mean and 8x8 cov."""
        mean_pos = np.asarray(measurement, dtype=np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def _mean_8_to_5(self, mean_8):
        x, y, a, h, vx, vy, va, vh = mean_8
        v, phi = _vphi_from_xy(vx, vy)
        return np.array([x, y, v, phi, 0.0], dtype=np.float64)

    def _cov_8_to_5(self, mean_8, cov_8):
        v, phi = _vphi_from_xy(mean_8[4], mean_8[5])
        cov_5 = np.eye(5, dtype=np.float64) * 1e-4
        cov_5[0:2, 0:2] = cov_8[0:2, 0:2]
        J_inv = np.array([
            [np.cos(phi), np.sin(phi)],
            [-np.sin(phi) / (v + 1e-8), np.cos(phi) / (v + 1e-8)]
        ])
        cov_5[2:4, 2:4] = J_inv @ cov_8[4:6, 4:6] @ J_inv.T
        cov_5[0:2, 2:4] = cov_8[0:2, 4:6] @ J_inv.T
        cov_5[2:4, 0:2] = cov_5[0:2, 2:4].T
        return cov_5

    def _mean_cov_5_to_8(self, mean_5, cov_5, mean_8_prev, cov_8_prev):
        x, y, v, phi, omega = mean_5
        vx, vy = _xy_from_vphi(v, phi)
        dt = self._dt
        a = mean_8_prev[2] + mean_8_prev[6] * dt
        h = mean_8_prev[3] + mean_8_prev[7] * dt
        va, vh = mean_8_prev[6], mean_8_prev[7]
        mean_8 = np.array([x, y, a, h, vx, vy, va, vh], dtype=np.float64)

        J = np.eye(8)
        J[4, 2] = np.cos(phi)
        J[4, 3] = -v * np.sin(phi)
        J[5, 2] = np.sin(phi)
        J[5, 3] = v * np.cos(phi)
        cov_xy_4 = np.eye(4)
        cov_xy_4[0:2, 0:2] = cov_5[0:2, 0:2]
        cov_xy_4[2:4, 2:4] = J[4:6, 2:4] @ cov_5[2:4, 2:4] @ J[4:6, 2:4].T
        cov_xy_4[0:2, 2:4] = cov_5[0:2, 2:4] @ J[4:6, 2:4].T
        cov_xy_4[2:4, 0:2] = cov_xy_4[0:2, 2:4].T

        idx_ah = [2, 3, 6, 7]
        motion_ah = np.eye(4)
        motion_ah[0, 2] = dt
        motion_ah[1, 3] = dt
        std_ah = [
            self._std_weight_position * mean_8_prev[3],
            self._std_weight_position * mean_8_prev[3],
            1e-5,
            self._std_weight_velocity * mean_8_prev[3],
        ]
        q_ah = np.diag(np.square(std_ah))
        cov_ah_4 = motion_ah @ cov_8_prev[np.ix_(idx_ah, idx_ah)] @ motion_ah.T + q_ah

        cov_8 = np.eye(8)
        cov_8[0:4, 0:4] = cov_xy_4
        cov_8[np.ix_(idx_ah, idx_ah)] = cov_ah_4
        return mean_8, cov_8

    def predict(self, mean, covariance, other_track_means=None):
        """Predict 8D state. other_track_means: optional list of 8D means (only [:, :2] used for social force)."""
        mean_8 = np.asarray(mean, dtype=np.float64)
        cov_8 = np.asarray(covariance, dtype=np.float64)
        mean_5 = self._mean_8_to_5(mean_8)
        cov_5 = self._cov_8_to_5(mean_8, cov_8)
        other_5 = [m[:2] for m in other_track_means] if other_track_means else None
        mean_5, cov_5 = self._ekf.predict(mean_5, cov_5, other_track_means=other_5)
        return self._mean_cov_5_to_8(mean_5, cov_5, mean_8, cov_8)

    def project(self, mean, covariance):
        """Project state to measurement space (x, y, a, h)."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean_proj = mean[:4].copy()
        cov_proj = covariance[:4, :4].copy() + innovation_cov
        return mean_proj, cov_proj

    def update(self, mean, covariance, measurement):
        """Update with 4D measurement (x, y, a, h)."""
        z_xy = np.asarray(measurement[:2], dtype=np.float64)
        mean_5 = self._mean_8_to_5(mean)
        cov_5 = self._cov_8_to_5(mean, covariance)
        mean_5, cov_5 = self._ekf.update(mean_5, cov_5, z_xy)
        vx, vy = _xy_from_vphi(mean_5[2], mean_5[3])
        a_new, h_new = measurement[2], measurement[3]
        va, vh = mean[6], mean[7]
        new_mean = np.array([mean_5[0], mean_5[1], a_new, h_new, vx, vy, va, vh], dtype=np.float64)
        v, phi = mean_5[2], mean_5[3]
        J_vy = np.array([[np.cos(phi), -v * np.sin(phi)], [np.sin(phi), v * np.cos(phi)]])
        new_cov = np.eye(8)
        new_cov[0:2, 0:2] = cov_5[0:2, 0:2]
        new_cov[4:6, 4:6] = J_vy @ cov_5[2:4, 2:4] @ J_vy.T
        new_cov[0:2, 4:6] = cov_5[0:2, 2:4] @ J_vy.T
        new_cov[4:6, 0:2] = new_cov[0:2, 4:6].T
        new_cov[2:4, 2:4] = np.diag(np.square([1e-2, self._std_weight_position * h_new]))
        new_cov[6:8, 6:8] = np.diag(np.square([1e-5, self._std_weight_velocity * h_new]))
        return new_mean, new_cov

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Squared Mahalanobis distance in measurement space."""
        mean_proj, cov_proj = self.project(mean, covariance)
        if only_position:
            mean_proj = mean_proj[:2]
            cov_proj = cov_proj[:2, :2]
            measurements = np.asarray(measurements)[:, :2]
        else:
            measurements = np.asarray(measurements)
        # Ensure positive-definiteness: symmetrise then clamp eigenvalues
        cov_proj = 0.5 * (cov_proj + cov_proj.T)
        eigvals, eigvecs = np.linalg.eigh(cov_proj)
        eigvals = np.clip(eigvals, 1e-2, None)
        cov_proj = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cholesky_factor = np.linalg.cholesky(cov_proj)
        d = measurements - mean_proj
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        return np.sum(z * z, axis=0)