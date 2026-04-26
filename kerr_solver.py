from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List

@dataclass
class State:
    t: float
    r: float
    theta: float
    phi: float
    pt: float
    pr: float
    ptheta: float
    pphi: float

    def convert_state_to_vector(self) -> List[float]:
        return [self.t, self.r, self.theta, self.phi, self.pt, self.pr, self.theta, self.pphi]

    @staticmethod
    def convert_vector_to_state(vector: List[float]) -> State:
        if len(vector) != 8:
            raise ValueError(f"Expected 8 values, but got {len(vector)}.")
        
        for index, value in enumerate(vector):
            if not isinstance(value, (int, float)):
                raise TypeError(f"Element at index {index} is not a number: {value!r}.")
        
        return State(*vector)

# ---------------- Physics ----------------

def delta(radius: float, black_hole_mass: float, black_hole_spin: float) -> float:
    return radius ** 2.0 - 2.0 * black_hole_mass * radius + black_hole_spin ** 2.0

def sigma(radius: float, theta: float, black_hole_spin: float) -> float:
    return radius ** 2.0 + (black_hole_spin * math.cos(theta)) ** 2.0

def metric_inverse(radius: float, theta: float, black_hole_mass: float, black_hole_spin: float) -> List[List[float]]:
    d: float = delta(radius, black_hole_mass, black_hole_spin)
    s: float = sigma(radius, theta, black_hole_spin)
    sin2: float = math.sin(theta) ** 2.0

    gtt: float = -1.0 * ((radius ** 2.0 + black_hole_spin ** 2.0) ** 2.0 - black_hole_spin ** 2.0 * d * sin2) / (d * s)
    gtphi: float = -2.0 * black_hole_mass * black_hole_spin * radius / (d * s)
    gphiphi: float = (d - sin2 * black_hole_spin ** 2.0) / (d * s * sin2)
    grr: float = d / s
    gthetatheta: float = 1.0 / s
    return [
        [gtt, 0, 0, gtphi],
        [0, grr, 0, 0],
        [0, 0, gthetatheta, 0],
        [gtphi, 0, 0, gphiphi]
    ]

def hamiltonian(state: State, black_hole_mass: float, black_hole_spin: float) -> float:
    ginv: List[List[float]] = metric_inverse(state.r, state.theta, black_hole_mass, black_hole_spin)
    mom: List[float] = [state.pt, state.pr, state.ptheta, state.pphi]
    h: float = 0.0
    for i in range(4):
        for j in range(4):
            h += ginv[i][j] * mom[i] * mom[j]
    h *= 0.5
    return h

def dh_dxi(i: int, state: State, tolerance: float, black_hole_mass: float, black_hole_spin: float) -> float:
    vector: List[float] = state.convert_state_to_vector()
    vector[i] += tolerance
    hp: float = hamiltonian(State.convert_vector_to_state(vector), black_hole_mass, black_hole_spin)
    vector[i] -= 2.0 * tolerance
    hm: float = hamiltonian(State.convert_vector_to_state(vector), black_hole_mass, black_hole_spin)
    return (hp - hm) / (2.0 * tolerance)

def derivatives(state: State, black_hole_mass: float, black_hole_spin: float) -> List[float]:
    ginv: List[List[float]] = metric_inverse(state.r, state.theta, black_hole_mass, black_hole_spin)
    mom: List[float] = [state.pt, state.pr, state.ptheta, state.pphi]
    eps: float = 1e-5

    dx: List[float] = [sum(ginv[i][j] * mom[j] for j in range(4)) for i in range(4)]
    dp: List[float] = [-dh_dxi(i, state, eps, black_hole_mass, black_hole_spin) for i in range(4)]

    return dx + dp
    
# ---------------- RK4 ----------------

def rk4_step_vector(function: callable[[State, float, float], List[float]],
             y: List[float],
             black_hole_mass: float,
             black_hole_spin: float,
             integration_step: float
             ) -> List[float]:
    k1 = function(State.convert_vector_to_state(y), black_hole_mass, black_hole_spin)
    
    y2: List[float] = [y[i] + 0.5 * integration_step * k1[i] for i in range(len(y))]
    k2 = function(State.convert_vector_to_state(y2), black_hole_mass, black_hole_spin)

    y3: List[float] = [y[i] + 0.5 * integration_step * k2[i] for i in range(len(y))]
    k3 = function(State.convert_vector_to_state(y3), black_hole_mass, black_hole_spin)

    y4: List[float] = [y[i] + 0.5 * integration_step * k3[i] for i in range(len(y))]
    k4 = function(State.convert_vector_to_state(y4), black_hole_mass, black_hole_spin)

    return [
        y[i] + (integration_step / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(len(y))
    ]

def check_if_hit_disk(state: State, inner_disk_radius: float, outer_disk_radius: float, tolerance: float = 1e-2) -> bool:
    # now disk is located at theta = pi / 2
    # in future I will add height to the disk so this function will require change
    if abs(state.theta - math.pi / 2.0) < tolerance:
        if inner_disk_radius <= state.r <= outer_disk_radius:
            return True
    return False

def integrate(state: State,
              black_hole_mass: float,
              black_hole_spin: float,
              number_of_steps: int,
              initial_integration_step: float,
              tolerance: float = 1e-5,
              minimal_integration_step: float = 1e-5,
              maximal_integration_step: float = 1.0,
              inner_disk_radius: float | None = None,
              outer_disk_radius: float | None = None
              ) -> List[State]:
    
    """Adaptive RK4 using step-doubling error estimate."""
    trajectory: List[State] = [state]
    y: List[float] = state.convert_state_to_vector()
    integration_step = initial_integration_step

    for _ in range(number_of_steps):
        current_state = State.convert_vector_to_state(y)
        # Check for disk collision
        if inner_disk_radius is not None and outer_disk_radius is not None:
            if check_if_hit_disk(current_state, inner_disk_radius, outer_disk_radius):
                print(f"Particle hits the disk - stopping integration")
                break

        # Full step 
        y_full: List[float] = rk4_step_vector(derivatives, y, black_hole_mass, black_hole_spin, integration_step)

        # Two half steps
        y_half: List[float] = rk4_step_vector(derivatives, y, black_hole_mass, black_hole_spin, integration_step / 2.0)
        y_half_2: List[float] = rk4_step_vector(derivatives, y_half, black_hole_mass, black_hole_spin, integration_step / 2.0)

        # Error estimate
        error: float = max(abs(y_full[i] - y_half_2[i]) for i in range(len(y)))

        if error < tolerance or integration_step <= minimal_integration_step:
            # Accept Step
            y = y_half_2
            trajectory.append(State.convert_vector_to_state(y))

            # Increase step if error is to small
            if error == 0:
                scale = 2.0
            else: 
                scale = 0.9 * (tolerance / error) ** 0.2
            
            integration_step = min(maximal_integration_step, integration_step * scale)
        else:
            # Reject step and reduce h
            scale = max(0.1, 0.9 * (tolerance / error) ** 0.25)
            integration_step = max(minimal_integration_step, integration_step * scale)
    
    return trajectory

if __name__ == "__main__":
    black_hole_mass: float = 1.0
    black_hole_spin: float = 0.9
    initial_integration_step: float = 0.1
    number_of_points_for_trajectory: int = 10000
    inner_disk_radius: float = 8.0
    outer_disk_radius: float = 20.0

    initial_state = State(
        t=0.0,
        r=50.0,
        theta=math.pi/3.3,
        phi=1.8,
        pt=-1.0,
        pr=0.0,
        ptheta=0.1,
        pphi=3.0
    )

    trajectory: List[State] = integrate(initial_state, black_hole_mass, black_hole_spin, number_of_points_for_trajectory, initial_integration_step)
    print(trajectory)