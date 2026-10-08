"""冻结门槛回归：使用真实记录的副本，验证创新泄入和源码漂移会被拒绝。"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_b_adaptations as audit

RUN = audit.ROOT/'outputs/matrix/B_rotation_reuse_20260920_r01/native'


class BaselineAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'run'
        self.run.mkdir()
        for p in RUN.iterdir():
            if p.is_file(): (self.run/p.name).symlink_to(p)

    def replace(self, name, text):
        p = self.run/name
        p.unlink()
        p.write_text(text)

    def result(self):
        return audit.audit(self.run, self.root/'report')

    def test_current_B_without_BERT(self):
        r = self.result()
        self.assertTrue(r['freeze_allowed'])
        self.assertEqual(r['active_innovation_count'], 0)
        self.assertFalse(r['paper_absolute_performance_is_gate'])

    def test_new_packing_leak_rejected(self):
        name = 'run_metadata_party2.txt'
        self.replace(name, (self.run/name).read_text().replace('heterogeneous_head_grouping=0', 'heterogeneous_head_grouping=1'))
        r = self.result()
        self.assertFalse(r['freeze_allowed'])
        self.assertIn('heterogeneous_head_grouping', r['active_innovations'])

    def test_changed_head_plan_rejected(self):
        name = 'head_plan_party1.csv'
        text = (self.run/name).read_text().replace('softmax_mpc,1.000000000', 'relu_norm_mpc,1.000000000', 1)
        self.replace(name, text)
        r = self.result()
        self.assertFalse(r['freeze_allowed'])
        self.assertIn('heterogeneous_heads', r['active_innovations'])

    def test_source_drift_requires_review_not_zero_claim(self):
        catalog = json.loads(audit.CATALOG.read_text())
        catalog['reviewed_source_sha256']['ours_vit_b16/runtime/edge_cloud_runtime.h'] = '0'*64
        path = self.root/'catalog.json'
        path.write_text(json.dumps(catalog))
        with patch.object(audit, 'CATALOG', path):
            r = self.result()
        self.assertFalse(r['freeze_allowed'])
        self.assertIsNone(r['active_innovation_count'])


if __name__ == '__main__':
    unittest.main()
