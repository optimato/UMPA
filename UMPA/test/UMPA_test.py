"""
Testing UMPA
"""

import unittest

from UMPA import match, match_unbiased
from UMPA import utils as u


class UMPATest(unittest.TestCase):

    def setUp(self):
        s = u.prep_simul()
        self.T = s['T']
        self.dx = s['dx']
        self.dy = s['dy']
        self.positions = s['positions']
        self.ref = s['ref']
        self.meas = s['meas']

    def tearDown(self):
        pass

    def test_UMPA(self):
        result = match(self.meas, self.ref, Nw=1, step=10)

    def test_UMPA_unbiased(self):
        result = match_unbiased(self.meas, self.ref, Nw=1, step=10)

