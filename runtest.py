# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:33:37 2015

@author: Administrator
"""

from constants import COV
from fitlif import LifModel
import unittest
import pickle
import copy
import numpy as np


class TestLif(unittest.TestCase):

    def setUp(self):
        with open('./pickles/test_lesniak_vs_lif_data.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        for key, item in data_dict.items():
            setattr(self, key, item)
        self.lifModel = LifModel(**self.fiber_rcv)
        self.mcnc_grouping = np.array([8, 5, 3, 1])
        self.trans_param_lesniak = copy.deepcopy(self.trans_param)
        self.trans_param_lesniak[:2] /= max(self.mcnc_grouping)

    def test_equiv(self):
        lif_response = self.lifModel.trans_param_to_predicted_fr(
            self.quantity_dict_list, self.trans_param)
        lesniak_response = self.lifModel.trans_param_to_predicted_fr(
            self.quantity_dict_list, self.trans_param_lesniak,
            model='Lesniak', mcnc_grouping=self.mcnc_grouping)
        self.assertTrue(np.allclose(
            lif_response, lesniak_response))

    def test_std(self):
        std = self.lifModel.fit_noise(
            self.trans_param_lesniak, self.quantity_dict_list, COV,
            model='Lesniak', mcnc_grouping=self.mcnc_grouping)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
