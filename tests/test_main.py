"""Main module"""

from unittest import TestCase, main

import torch

from wheresmycar.main import zeros


class MainTest(TestCase):
    """Test Main"""

    def test_none(self):
        """Test is None"""
        self.assertIsNone(None)

    def test_zeros(self):
        """Test zeros"""
        result = zeros()
        self.assertIsInstance(result, torch.Tensor)


if __name__ == "__main__":
    main()
