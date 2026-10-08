import copy
import json
import tempfile
import unittest
from pathlib import Path
from run_research2 import DEFAULT_CONFIG, ROOT, load_config, secure_command, offline_command, edge_cloud_metadata_agrees
from version_matrix import MATRIX, plan_binding, file_hash
from experiment_contract import CONTRACT, CONTRACT_SHA256, ONLINE_INVARIANTS

class SecureConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.config = load_config(DEFAULT_CONFIG)
        self.plan = self.folder / 'plan.txt'
        self.plan.write_text((ROOT / 'ours_vit_b16/experiments/all_softmax_head_plan.txt').read_text())
        for section, keys in [('paths', ('model', 'model_manifest', 'preprocess_manifest')),
                              ('secure', ('input', 'binary_override', 'phantom_library_override'))]:
            for key in keys:
                p = self.folder / key
                p.write_text(key)
                self.config[section][key] = str(p)

    def paired(self, parent='HA'):
        self.config['secure'].update(version=parent, head_plan=str(self.plan))
        path = self.folder / 'config.secure.json'
        path.write_text(json.dumps(self.config))
        binding = plan_binding(self.plan)
        binding['experiment_contract'] = CONTRACT
        binding['experiment_contract_sha256'] = CONTRACT_SHA256
        binding['image_id'] = 'sha256:' + 'c' * 64
        binding['files'] = {'paths.model': file_hash(self.config['paths']['model'])}
        path.with_name('plan_binding.json').write_text(json.dumps(binding))
        self.config['secure'].update(version='HAP' if parent == 'HA' else 'HP', paired_config=str(path))

    def test_edge_cloud_audit_requires_real_matching_hashes(self):
        rows = [{'edge_cloud_role': role, 'edge_cloud_plan_agreed': '1',
                 'edge_cloud_contract_sha256': 'a' * 64} for role in ('cloud', 'edge')]
        self.assertTrue(edge_cloud_metadata_agrees(rows))
        for row in rows:
            row['edge_cloud_contract_shgrouped56'] = row.pop('edge_cloud_contract_sha256')
        self.assertFalse(edge_cloud_metadata_agrees(rows))
        for value in ('', 'not-a-hash', 'z' * 64):
            for row in rows: row['edge_cloud_contract_sha256'] = value
            self.assertFalse(edge_cloud_metadata_agrees(rows))
        rows[0]['edge_cloud_contract_sha256'] = 'a' * 64
        rows[1]['edge_cloud_contract_sha256'] = 'b' * 64
        self.assertFalse(edge_cloud_metadata_agrees(rows))

    def test_seven_versions_have_one_dispatch(self):
        for version in MATRIX:
            config = copy.deepcopy(self.config)
            config['secure'].update(version=version, head_plan=None if version in ('B','BP') else str(self.plan))
            if version in ('HP','HAP'):
                self.paired('H' if version == 'HP' else 'HA')
                config = copy.deepcopy(self.config)
            command, env = secure_command(config)
            self.assertEqual(command[2], version)
            self.assertNotIn('OURS_VIT_GROUP_PACKING', env)
            self.assertEqual(command[-2:], ['12','full'])

    def test_old_names_rejected(self):
        for name in ('a0','a1','a2','a3','a4','native','selective'):
            config = copy.deepcopy(self.config)
            config['secure']['version'] = name
            with self.assertRaises(ValueError): secure_command(config)

    def test_obsolete_fields_rejected(self):
        path = self.folder / 'old.json'
        for field in ('ablation','group_packing'):
            path.write_text(json.dumps({'secure': {field: 0}}))
            with self.assertRaises(ValueError): load_config(path)

    def test_paired_plan_cannot_change_protocol_or_alpha(self):
        self.paired()
        secure_command(self.config)
        self.plan.write_text(self.plan.read_text().replace('0,0,0,softmax_mpc,1', '0,0,0,he_linear,0.5'))
        with self.assertRaisesRegex(ValueError, 'source plan'): secure_command(self.config)

    def test_paired_source_cannot_change_definitions(self):
        self.paired()
        path = self.folder / 'plan_binding.json'
        binding = json.loads(path.read_text())
        binding['experiment_contract']['functions']['R'] = 'relu without normalization'
        path.write_text(json.dumps(binding))
        with self.assertRaisesRegex(ValueError, 'paired source experiment definitions'):
            secure_command(self.config)

    def test_paired_execution_configuration_is_frozen(self):
        self.paired()
        self.config['secure']['gpu_linear_streams'] = 4
        with self.assertRaisesRegex(ValueError, 'gpu_linear_streams'): secure_command(self.config)

    def test_paired_checkpoint_in_place_change_is_rejected(self):
        self.paired()
        Path(self.config['paths']['model']).write_text('changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'frozen asset'): secure_command(self.config)

    def test_pairs_require_source_snapshot(self):
        self.config['secure'].update(version='HAP', head_plan=str(self.plan))
        with self.assertRaisesRegex(ValueError, 'paired_config'): secure_command(self.config)

    def test_search_is_only_ha_or_j(self):
        for version in MATRIX:
            self.config['secure']['version'] = version
            if version in ('HA','J'):
                command = offline_command(self.config)
                self.assertEqual(command[command.index('--version')+1], version)
            else:
                with self.assertRaises(ValueError): offline_command(self.config)

    def test_semantics_ignore_physical_permutation(self):
        before = plan_binding(self.plan)
        lines = self.plan.read_text().splitlines()
        for i,line in enumerate(lines):
            parts = line.split(',')
            if len(parts)==5 and parts[0].isdigit():
                parts[1] = str(11-int(parts[1]));lines[i] = ','.join(parts)
        self.plan.write_text('\n'.join(lines)+'\n')
        after = plan_binding(self.plan)
        self.assertEqual(before['semantic_plan_sha256'], after['semantic_plan_sha256'])
        self.assertNotEqual(before['physical_plan_sha256'], after['physical_plan_sha256'])

    def test_assignment_does_not_override_version(self):
        assignment = self.folder / 'assignment.json'
        assignment.write_text(json.dumps({'runtime': {'version': 'HA','head_plan': str(self.plan)},
                                          'model_sha256': file_hash(self.config['paths']['model'])}))
        self.config['secure'].update(version='H', assignment=str(assignment))
        with self.assertRaisesRegex(ValueError, 'assignment version'): secure_command(self.config)

    def test_assignment_requires_offline_fixed_contract_and_frozen_plan(self):
        assignment = self.folder / 'assignment.json'
        document = {
            'runtime': {'version': 'HA', 'head_plan': str(self.plan), **plan_binding(self.plan)},
            'model_sha256': file_hash(self.config['paths']['model']),
            'experiment_contract': copy.deepcopy(CONTRACT),
            'experiment_contract_sha256': CONTRACT_SHA256,
            'online_invariants': dict(ONLINE_INVARIANTS),
        }
        self.config['secure'].update(version='HA', assignment=str(assignment))
        assignment.write_text(json.dumps(document))
        secure_command(self.config)
        for field in ONLINE_INVARIANTS:
            bad = copy.deepcopy(document)
            bad['online_invariants'].pop(field)
            assignment.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'offline-fixed contract'):
                secure_command(self.config)
        for field, value in [('dynamic_selection', True), ('fixed_public_assignment', False)]:
            bad = copy.deepcopy(document)
            bad['online_invariants'][field] = value
            assignment.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'offline-fixed contract'):
                secure_command(self.config)
        bad = copy.deepcopy(document)
        bad['experiment_contract']['security']['ckks_security_target'] = 'none'
        assignment.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'definitions'):
            secure_command(self.config)
        assignment.write_text(json.dumps(document))
        self.plan.write_text(self.plan.read_text().replace('0,0,0,softmax_mpc,1', '0,0,0,he_linear,0.5'))
        with self.assertRaisesRegex(ValueError, 'semantic plan mismatch'):
            secure_command(self.config)

    def test_baselines_reject_custom_plan(self):
        for version in ('B','BP'):
            self.config['secure'].update(version=version, head_plan=str(self.plan))
            with self.assertRaises(ValueError): secure_command(self.config)

    def test_output_path_and_gpu_pair(self):
        self.config['secure'].update(gpu_pair=[1,7], output_dir=str(ROOT/'outputs/test/native'))
        command, env = secure_command(self.config)
        self.assertEqual(command[4], 'outputs/test/native')
        self.assertEqual(env['OURS_VIT_PHYSICAL_GPUS'], '1,7')
        self.config['secure']['output_dir'] = '/tmp/outside'
        with self.assertRaises(ValueError): secure_command(self.config)

if __name__ == '__main__': unittest.main()
