# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:33:37 2015

@author: Administrator
"""

from fitlif import LifModel
import unittest
import pickle
import copy
import numpy as np


class TestLesniakVsLif(unittest.TestCase):

    def setUp(self):
        with open('./pickles/test_lesniak_vs_lif_data.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        for key, item in data_dict.items():
            setattr(self, key, item)
        self.lifModel = LifModel(**self.fiber_rcv)

    def test_equiv(self):
        mcnc_grouping = [8, 5, 3, 1]
        lif_response = self.lifModel.trans_param_to_predicted_fr(
            self.quantity_dict_list, self.trans_param)
        trans_param_lesniak = copy.deepcopy(self.trans_param)
        trans_param_lesniak[:2] /= max(mcnc_grouping)
        lesniak_response = self.lifModel.trans_param_to_predicted_fr(
            self.quantity_dict_list, trans_param_lesniak,
            model='Lesniak', mcnc_grouping=mcnc_grouping)
        print(lesniak_response)
        self.assertTrue(np.allclose(
            lif_response, lesniak_response))

if __name__ == '__main__':
    unittest.main()
