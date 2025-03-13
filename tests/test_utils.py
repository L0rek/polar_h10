import unittest

from polar_h10.core import utils as polar_utils


class TestUtils(unittest.TestCase):
    def test_param_serialization(self):
        params = {"SAMPLE_RATE": [25, 50, 100, 200], "RESOLUTION": 16, "RANGE": [2, 4, 8]}
        ser_params = b"\x00\x04\x19\x002\x00d\x00\xc8\x00\x01\x01\x10\x00\x02\x03\x02\x00\x04\x00\x08\x00"

        self.assertEqual(polar_utils.serialize_pmd_parameters(params), ser_params)

    def test_param_deserialization(self):
        params = {"SAMPLE_RATE": [25, 50, 100, 200], "RESOLUTION": 16, "RANGE": [2, 4, 8]}
        ser_params = b"\x00\x04\x19\x002\x00d\x00\xc8\x00\x01\x01\x10\x00\x02\x03\x02\x00\x04\x00\x08\x00"

        self.assertEqual(polar_utils.deserialize_pmd_parameters(ser_params), params)

    def test_ecg_data_parser(self):
        samples = list(range(-1000, 1000, 100))
        frame = b"".join([x.to_bytes(3, "little", signed=True) for x in samples])

        _, result = polar_utils.process_ecg_samples(0, frame)
        self.assertEqual(result, {"lead_1": samples}, "frame_type_0")

    def test_acc_data_parser(self):
        samples_x = list(range(-1000, 1000, 100))
        samples_y = list(range(-1010, 990, 100))
        samples_z = list(range(-1020, 980, 100))

        frame = b""
        for axis in zip(samples_x, samples_y, samples_z):
            for val in axis:
                frame += val.to_bytes(2, "little", signed=True)

        _, result = polar_utils.process_acc_samples(1, frame)
        self.assertEqual(result, {"x": samples_x, "y": samples_y, "z": samples_z}, "frame_type_1")

        frame = b""
        for axis in zip(samples_x, samples_y, samples_z):
            for val in axis:
                frame += val.to_bytes(3, "little", signed=True)

        _, result = polar_utils.process_acc_samples(2, frame)
        self.assertEqual(result, {"x": samples_x, "y": samples_y, "z": samples_z}, "frame_type_2")

    def test_hr_data_parser(self):
        sample_0 = {"heart_rate": 84}
        sample_1 = {"heart_rate": 276, "sensor_contact": True, "rr_interval": [905]}
        sample_2 = {"heart_rate": 84, "sensor_contact": False, "rr_interval": [816, 842]}
        sample_3 = {"heart_rate": 84, "sensor_contact": True, "energy_expended": 73, "rr_interval": [905]}
        frame_0 = b"\x01\x54"
        frame_1 = b"\x17\x14\x01\x9f\x03"
        frame_2 = b"\x14\x54\x44\x03\x5f\x03"
        frame_3 = b"\x1e\x54\x49\x00\x9f\x03"

        result = polar_utils.process_hr_samples(frame_0)
        self.assertEqual(result, sample_0, "sample_0")

        result = polar_utils.process_hr_samples(frame_1)
        self.assertEqual(result, sample_1, "sample_1")

        result = polar_utils.process_hr_samples(frame_2)
        self.assertEqual(result, sample_2, "sample_2")

        result = polar_utils.process_hr_samples(frame_3)
        self.assertEqual(result, sample_3, "sample_3")


if __name__ == "__main__":
    unittest.main()
