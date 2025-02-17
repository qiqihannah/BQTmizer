import unittest
from bootqa import Sampler, QA_solver
import pandas as pd

class BQTmizer_Test(unittest.TestCase):
    def setUp(self):
        data_csv = pd.read_csv("validation/dataset/PaintControl_TCM.csv", dtype={"time": float, "rate": float})
        self.data = data_csv.drop(data_csv[data_csv['rate'] == 0].index)
    def test_bootstrap_sampling(self):
        sampler = Sampler(len(self.data), 0.1, 5)
        subproblem_num = sampler.random_sample_unique()
        sample_total_list = sampler.bootstrap_sampling(subproblem_num)

        self.assertEqual(subproblem_num, 1)
        self.assertEqual(len(sample_total_list), subproblem_num)

    def test_run_bqtmizer(self):
        weights = {"time": 1 / 3, "rate": 1 / 3, "num": 1 / 3}
        qa_solver = QA_solver(7, self.data, ["rate"], ["time"], weights)
        sample_first_list, qpu_access_list, qubit_avg, log_df, sampleset_list = qa_solver.run_qpu([[33, 2, 85, 72, 81]], 7, "...")
        self.assertIsInstance(sample_first_list, list)  # Should be a list
        self.assertIsInstance(qpu_access_list, list)  # Should be a list
        self.assertIsInstance(qubit_avg, (int, float))  # Should be a number
        self.assertIsInstance(log_df, pd.DataFrame)  # Assuming log_df is a Pandas DataFrame
        self.assertIsInstance(sampleset_list, list)  # Should be a list


if __name__ == "__main__":
    unittest.main()
