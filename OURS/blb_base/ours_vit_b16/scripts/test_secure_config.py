import json
import tempfile
import unittest
from pathlib import Path

from run_research2 import DEFAULT_CONFIG, ROOT, load_config, secure_command, secure_environment


class SecureConfigurationTest(unittest.TestCase):
    def test_a0_disables_heterogeneity_without_unpacked_validation_path(self):
        config = load_config(DEFAULT_CONFIG)
        config['secure'].update(ablation='a0', assignment=None, head_plan=None,
                                group_packing=0)
        command, env = secure_command(config)
        self.assertEqual(command[2], 'a0')
        self.assertEqual(env['OURS_VIT_GROUP_PACKING'], '1')
        self.assertNotIn('OURS_VIT_BASELINE_ATTENTION_PACKING', env)
        self.assertEqual(env['OURS_VIT_FORMAL_A0_BASELINE'], '1')
        self.assertEqual(config['secure']['group_packing'], 0)
        self.assertEqual(secure_environment(config)['OURS_VIT_GROUP_PACKING'], '0')

    def test_a4_retains_packing_without_conversion_modes(self):
        config = load_config(DEFAULT_CONFIG)
        config['secure'].update(ablation='a4', head_plan='public_plan.txt',
                                group_packing=0)
        command, env = secure_command(config)
        self.assertEqual(command[2], 'a4')
        self.assertEqual(env['OURS_VIT_GROUP_PACKING'], '0')
        self.assertNotIn('conversion_mode', config['secure'])
        self.assertNotIn('OURS_VIT_CONVERSION_MODE', env)
        self.assertNotIn('OURS_VIT_FORMAL_A0_BASELINE', env)

    def test_assignment_effective_a0_overrides_shared_a4_settings(self):
        config = load_config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            assignment = Path(directory) / 'assignment.json'
            assignment.write_text(json.dumps({'runtime': {'ablation': 'A0', 'head_plan': None}}))
            config['secure'].update(ablation='a4', assignment=str(assignment), group_packing=0)
            command, env = secure_command(config)
        self.assertEqual(command[2], 'a0')
        self.assertEqual(env['OURS_VIT_GROUP_PACKING'], '1')

    def test_requested_gpu_pair_is_not_replaced(self):
        config = load_config(DEFAULT_CONFIG)
        config['secure']['gpu_pair'] = [1, 7]
        _, env = secure_command(config)
        self.assertEqual(env['OURS_VIT_PHYSICAL_GPUS'], '1,7')
        self.assertEqual(env['OURS_VIT_ALLOW_SHARED_GPU'], '0')

    def test_single_gpu_remains_supported(self):
        config = load_config(DEFAULT_CONFIG)
        config['secure'].update(gpu_id=7, gpu_pair=None)
        _, env = secure_command(config)
        self.assertEqual(env['OURS_VIT_PHYSICAL_GPUS'], '7,7')
        self.assertEqual(env['OURS_VIT_ALLOW_SHARED_GPU'], '1')

    def test_absolute_output_is_translated_for_container_runner(self):
        config = load_config(DEFAULT_CONFIG)
        config['secure']['output_dir'] = str(ROOT / 'outputs/test/native')
        command, _ = secure_command(config)
        self.assertEqual(command[4], 'outputs/test/native')

    def test_output_outside_outputs_is_rejected(self):
        config = load_config(DEFAULT_CONFIG)
        config['secure']['output_dir'] = '/tmp/not-an-experiment'
        with self.assertRaises(ValueError):
            secure_command(config)


if __name__ == '__main__':
    unittest.main()
