"""Regression for the rigid-pose contract at the MPM projection boundary."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from robotic_cutting_sim.newton_probe import NewtonTestbed


class Array:
    def __init__(self,data):
        self.data=np.array(data,dtype=float)
    def assign(self,data):
        self.data=np.array(data)
    def numpy(self):
        return self.data.copy()


def test_projection_uses_current_collider_after_each_buffer_swap():
    def state():
        return SimpleNamespace(body_q=Array([[0,0,0,0,0,0,1]]),body_qd=Array([[0]*6]),
                               particle_q=Array([[.03,0,.01]]))
    bed=NewtonTestbed.__new__(NewtonTestbed)
    bed.wp=SimpleNamespace(ScopedDevice=lambda _:nullcontext())
    bed.device="cpu"; bed.state=state(); bed.next_state=state(); bed.half_extents=np.array([.001,.01,.01])
    bed.collider_bodies=np.array([0]); bed.blade=0
    expected={}
    def project(input_state,output_state,dt):
        np.testing.assert_allclose(input_state.body_q.numpy()[0],expected["pose"],atol=1e-7)
        np.testing.assert_allclose(input_state.body_qd.numpy()[0],expected["twist"],atol=1e-7)
    # MPM intentionally does not copy body arrays into output_state.
    bed.solver=SimpleNamespace(step=lambda *args:None,project_outside=project,
        collect_collider_impulses=lambda _: (Array([[0,0,0]]),Array([[0,0,0]]),SimpleNamespace(numpy=lambda:np.array([0],int))))
    for x in [-.05,.04,-.02]:
        expected["pose"]=np.array([x,0,.03,0,0,0,1]); expected["twist"]=np.array([.1,0,0,0,0,0])
        bed.step(.001,expected["pose"],expected["twist"])
