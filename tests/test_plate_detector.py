"""Main module"""
import os

from unittest import TestCase, main, mock

from wheresmycar import PlateDetector


class PlateDetectorTest(TestCase):
    """Test PlateDetector class
    """
    DIRNAME = os.path.dirname(__file__)
    TEST_IMAGES_DIRECTORY = os.path.join(DIRNAME, "data/images/")

    def setUp(self):
        self.pd = PlateDetector(False)

    def test_get_device(self):
        """
        Test device with cuda disabled.
        Expect: returns PlateDetector.DEVICE_CPU
        """
        device = self.pd.get_device(False)
        self.assertEqual(device, PlateDetector.DEVICE_CPU)

    @mock.patch("wheresmycar.plate_detector.torch.cuda.is_available")
    def test_get_device_with_cuda(self, mock_is_avaiable):
        """
        Test device with cuda enabled.
        Expect: returns PlateDetector.DEVICE_CUDA
        """
        mock_is_avaiable.return_value = True
        device = self.pd.get_device(True)
        self.assertEqual(device, PlateDetector.DEVICE_CUDA)

    @mock.patch("wheresmycar.plate_detector.torch.cuda.is_available")
    def test_get_device_without_cuda(self, mock_is_avaiable):
        """
        Test device with cuda enabled.
        Expect: returns PlateDetector.DEVICE_CPU
        """
        mock_is_avaiable.return_value = False
        device = self.pd.get_device(True)
        self.assertEqual(device, PlateDetector.DEVICE_CPU)

    def test_detect(self):
        """
        Test detect.
        Excpet: detects 1 number plate on each test image
        """
        results = self.pd.detect(self.TEST_IMAGES_DIRECTORY)
        self.assertEqual(len(results), 2)
        for r in results:
            length = len(r.boxes)
            self.assertEqual(length, 1)


if __name__ == "__main__":
    main()
