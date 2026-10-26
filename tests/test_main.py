import unittest
from main import validate_epochs

class EpochTests(unittest.TestCase):
    def test_positive_epoch_count(self): self.assertEqual(validate_epochs(3), 3)
    def test_zero_is_rejected(self):
        with self.assertRaises(ValueError): validate_epochs(0)

if __name__ == "__main__": unittest.main()
