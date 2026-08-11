import unittest, torch
from icefall_wavlm.frontend import wavlm_output_lengths

class TestLengths(unittest.TestCase):
    def test_wavlm_base_one_second(self):
        # WavLM/Wav2Vec2-style kernels/strides produce ~49 frames for 16000 samples.
        kernels=[10,3,3,3,3,2,2]; strides=[5,2,2,2,2,2,2]
        n=wavlm_output_lengths(torch.tensor([16000]), kernels, strides)
        self.assertEqual(int(n[0]), 49)
