import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reliability_teacher_v3 import may_reserve,reservation_usd


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.protocol={'budget_usd':1.0,'max_generation_requests':12,'max_input_allowance':24000,'max_output_tokens':1536,'reserved_rates_per_million':{'input':1.5,'output':7.5}}

    def test_unknown_timeout_reservation_is_not_released(self):
        self.assertFalse(may_reserve([{'reserved_usd':0.96}],0.05,self.protocol))

    def test_request_cap_cannot_be_bypassed_by_low_cost(self):
        self.assertFalse(may_reserve([{'reserved_usd':0.001} for _ in range(12)],0.001,self.protocol))

    def test_oversized_payload_stops_before_request(self):
        with self.assertRaises(ValueError):reservation_usd({'text':'x'*24000},self.protocol)

    def test_reasoning_and_output_allowance_reserved(self):
        cost=reservation_usd({'text':'hello'},self.protocol)
        self.assertGreaterEqual(cost,1536*7.5/1e6)
        self.assertTrue(may_reserve([],cost,self.protocol))


if __name__=='__main__':unittest.main()
