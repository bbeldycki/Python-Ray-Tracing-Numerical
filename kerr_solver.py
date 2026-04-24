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