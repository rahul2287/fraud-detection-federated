import unittest
import pandas as pd

class TestDataIntegrity(unittest.TestCase):
    def test_data_format(self):
        df = pd.read_csv("../data/synthetic_data.csv")
        self.assertIn("amount", df.columns)
        self.assertIn("is_fraud", df.columns)
        self.assertEqual(df["is_fraud"].dropna().isin([0, 1]).all(), True)

if __name__ == "__main__":
    unittest.main()
