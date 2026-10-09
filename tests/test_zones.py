"""Zone / tripwire tests with synthetic track stubs."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.zones import Zone, Tripwire


class StubTrack:
    _id = 0

    def __init__(self, centers):
        StubTrack._id += 1
        self.id = StubTrack._id
        self._centers = list(centers)
        self._i = 0

    def center(self):
        return self._centers[min(self._i, len(self._centers) - 1)]

    def step(self):
        self._i += 1


def test_zone_enter_exit_and_dwell():
    z = Zone("door", [(0, 0), (100, 0), (100, 100), (0, 100)])
    t = StubTrack([(-10, 50), (50, 50), (50, 50), (150, 50)])
    evs = []
    for f in range(4):
        evs += z.update([t], f)
        t.step()
    kinds = [e[0] for e in evs]
    assert kinds == ["enter", "exit"], kinds
    assert z.entries == 1 and z.exits == 1
    assert z.dwell[t.id] == 2  # frames 1..3 -> entered at 1, exited at 3


def test_tripwire_direction():
    # Convention: stand at p1 looking toward p2; "forward" = crossing
    # from your left to your right. Wire goes down-screen, so the
    # viewer's left is +x (east): right->left crossing is forward.
    w = Tripwire("gate", (50, 0), (50, 100))  # directed downward
    t = StubTrack([(60, 50), (40, 50)])  # right -> left
    evs = []
    for _ in range(2):
        evs += w.update([t])
        t.step()
    assert evs == [("forward", t.id)], evs
    assert w.forward == 1 and w.backward == 0

    # And back left -> right = backward
    t2 = StubTrack([(40, 50), (60, 50)])
    evs = []
    for _ in range(2):
        evs += w.update([t2])
        t2.step()
    assert ("backward", t2.id) in evs


def test_zone_forgets_vanished_tracks():
    z = Zone("door", [(0, 0), (100, 0), (100, 100), (0, 100)])
    t = StubTrack([(50, 50)])
    z.update([t], 0)
    assert t.id in z._inside
    z.update([], 1)  # track vanished
    assert t.id not in z._inside
