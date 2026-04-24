import math
from dataclasses import dataclass
from typing import List, Iterable

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
        return [self.t, self.r, self.theta, self.phi,
                self.pt, self.pr, self.ptheta, self.pphi]

    @staticmethod
    def convert_vector_to_state(vector: Iterable[float]) -> State:
        try:
            values = list[vector]
        except TypeError:
            raise TypeError("Vector must be an iterable of 8 numberic values.")
        
        if len(values) != 8:
            raise ValueError(f"Expected 8 values, but got {len(values)}.")
        
        for index, value in enumerate(values):
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
    dp: List[float] = [dh_dxi(i, state, eps, black_hole_mass, black_hole_spin) for i in range(4)]

    return dx + dp
    

if __name__ == "__main__":
    black_hole_mass: float = 1.0
    black_hole_spin: float = 0.9
    initial_intergration_step: float = 0.1
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

    trajectory: List[float] = []