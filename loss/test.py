import unittest
from loss import zero_one_loss, cross_entropy_loss, softmax_multinomial_loss, hinge_loss, squared_l2_loss
import numpy as np
class LossTests(unittest.TestCase):

    def test_zero_one_loss(self):
        y_t = np.array([0,1,0,0,1,0])
        y_p = np.array([0,1,1,0,0, 0])
        diff = np.sum(np.abs(y_t - y_p))
        loss = diff/float(y_t.shape[0])
        self.assertAlmostEqual(loss, zero_one_loss(y_t, y_p))
        return True

    def test_cross_entropy_loss(self):
        pass

    def test_hinge_loss(self):
        pass

    def test_squared_loss(self):
        pass

    def test_softmax_multinomial_loss(self):
        pass

if __name__ == "__main__":
    unittest.main()
