#!/usr/bin/env python3
"""Test SOLO locali per supervisore senza timeout: rescue prima di DELETE."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gcp_paper_rf_supervisor as sup


class SupervisorTests(unittest.TestCase):
    def make_dest(self, folder):
        path = Path(folder)
        (path / 'campaign.json').write_text(json.dumps({
            'pin': sup.collector.PIN, 'schema': 6, 'cpu': 1,
            'target': 2, 'block': 2, 'entries': [{'name': 'hashing'}]
        }))
        return path

    def test_dry_run_never_deletes(self):
        with TemporaryDirectory() as temp:
            self.make_dest(temp)
            with patch.object(sup, 'print_local_plan'), patch.object(sup, 'delete_cluster') as delete:
                self.assertEqual(sup.main(['--dest', temp]), 0)
                delete.assert_not_called()

    def test_success_does_not_try_to_rescue_and_deletes(self):
        with TemporaryDirectory() as temp:
            self.make_dest(temp)
            actions = []
            with patch.object(sup, 'CAMPAIGN', Path(__file__)), \
                 patch.object(sup, 'supervise', side_effect=lambda _: actions.append('run')), \
                 patch.object(sup, 'emergency_salvage', side_effect=lambda _: actions.append('rescue')), \
                 patch.object(sup, 'delete_cluster', side_effect=lambda: actions.append('delete') or True):
                self.assertEqual(sup.main(['--dest', temp, '--run', '--confirm-delete']), 0)
            self.assertEqual(actions, ['run', 'delete'])

    def test_error_recover_rescue_before_delete(self):
        with TemporaryDirectory() as temp:
            self.make_dest(temp)
            actions = []
            with patch.object(sup, 'CAMPAIGN', Path(__file__)), \
                 patch.object(sup, 'supervise', side_effect=RuntimeError('errore')), \
                 patch.object(sup, 'try_recover_checkpoint', side_effect=lambda _: actions.append('recover')), \
                 patch.object(sup, 'emergency_salvage', side_effect=lambda _: actions.append('rescue') or True), \
                 patch.object(sup, 'delete_cluster', side_effect=lambda: actions.append('delete') or True):
                self.assertEqual(sup.main(['--dest', temp, '--run', '--confirm-delete']), 1)
            self.assertEqual(actions, ['recover', 'rescue', 'delete'])

    def test_rescue_failure_still_deletes(self):
        with TemporaryDirectory() as temp:
            self.make_dest(temp)
            actions = []
            with patch.object(sup, 'CAMPAIGN', Path(__file__)), \
                 patch.object(sup, 'supervise', side_effect=RuntimeError('errore')), \
                 patch.object(sup, 'try_recover_checkpoint'), \
                 patch.object(sup, 'emergency_salvage', side_effect=RuntimeError('rete giu')), \
                 patch.object(sup, 'delete_cluster', side_effect=lambda: actions.append('delete') or True):
                self.assertEqual(sup.main(['--dest', temp, '--run', '--confirm-delete']), 1)
            self.assertEqual(actions, ['delete'])

    def test_missing_manifest_refuses_without_gcp_or_rescue(self):
        with TemporaryDirectory() as temp:
            actions = []
            with patch.object(sup, 'try_recover_checkpoint', side_effect=lambda _: actions.append('recover')), \
                 patch.object(sup, 'emergency_salvage', side_effect=lambda _: actions.append('rescue') or False), \
                 patch.object(sup, 'delete_cluster', side_effect=lambda: actions.append('delete') or True):
                self.assertEqual(sup.main(['--dest', temp, '--run', '--confirm-delete']), 2)
            self.assertEqual(actions, [])
            self.assertFalse((Path(temp)/'emergency').exists())

    def test_malformed_manifest_refuses_without_gcp(self):
        with TemporaryDirectory() as temp:
            (Path(temp)/'campaign.json').write_text('{}')
            with patch.object(sup, 'CAMPAIGN', Path(sup.collector.__file__)), \
                 patch.object(sup, 'delete_cluster') as delete, \
                 patch.object(sup, 'emergency_salvage') as rescue:
                self.assertEqual(sup.main(['--dest', temp, '--run', '--confirm-delete']), 2)
                delete.assert_not_called()
                rescue.assert_not_called()

    def test_invalid_entries_refuses_without_gcp(self):
        with TemporaryDirectory() as temp:
            self.make_dest(temp)
            manifest = json.loads((Path(temp)/'campaign.json').read_text())
            manifest['entries'] = []
            (Path(temp)/'campaign.json').write_text(json.dumps(manifest))
            with patch.object(sup, 'CAMPAIGN', Path(sup.collector.__file__)), \
                 patch.object(sup, 'delete_cluster') as delete:
                self.assertEqual(sup.main(['--dest', temp, '--run', '--confirm-delete']), 2)
                delete.assert_not_called()

    def test_delete_failure_distinct_code(self):
        with TemporaryDirectory() as temp:
            self.make_dest(temp)
            with patch.object(sup, 'CAMPAIGN', Path(__file__)), \
                 patch.object(sup, 'supervise'), \
                 patch.object(sup, 'delete_cluster', return_value=False):
                self.assertEqual(sup.main(['--dest', temp, '--run', '--confirm-delete']), 3)

    def test_unsupported_time_limit_flags(self):
        with TemporaryDirectory() as temp:
            self.make_dest(temp)
            with self.assertRaises(SystemExit) as cm:
                sup.main(['--dest', temp, '--max-hours', '12'])
            self.assertEqual(cm.exception.code, 2)

    def test_completed_export_verifies_hashes(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            parts = {'paper-rf-dataset.csv': b'id\nabc\n',
                     'paper-rf-provenance.csv': b'id\nabc\n', 'campaign.json': b'{}'}
            for file, data in parts.items():
                (folder / file).write_bytes(data)
            manifest = {'pin': sup.collector.PIN, 'rows': 1,
                        'dataset_sha256': hashlib.sha256(parts['paper-rf-dataset.csv']).hexdigest(),
                        'provenance_sha256': hashlib.sha256(parts['paper-rf-provenance.csv']).hexdigest(),
                        'campaign_manifest_sha256': hashlib.sha256(parts['campaign.json']).hexdigest()}
            (folder / 'dataset_manifest.json').write_text(json.dumps(manifest))
            with patch.object(sup, 'phase_info', return_value=({}, 1, 1, None)):
                sup.verify_export(folder, 1)
                (folder / 'paper-rf-dataset.csv').write_bytes(b'corrupt')
                with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                    sup.verify_export(folder, 1)

    def test_snapshot_has_sha_and_keeps_partial_out_of_final(self):
        with TemporaryDirectory() as temp:
            destination = Path(temp) / 'raw.jsonl'
            data = b'{"schema_version":6}\n'
            sha = hashlib.sha256(data).hexdigest()
            with patch.object(sup, 'remote_ssh', side_effect=[sha, '/home/ale']), \
                 patch.object(sup, 'remote_scp', side_effect=lambda _h, _s, target: target.write_bytes(data)):
                self.assertEqual(sup.snapshot_remote('sl-arm-1', '/var/lib/serverledge/profiling-samples.jsonl',
                                                     destination, 'sl-arm-1_active.jsonl'), sha)
            self.assertEqual(destination.read_bytes(), data)
            self.assertFalse(destination.with_name('raw.jsonl.download').exists())

    def test_snapshot_sha_mismatch_not_promoted(self):
        with TemporaryDirectory() as temp:
            destination = Path(temp) / 'raw.jsonl'
            with patch.object(sup, 'remote_ssh', side_effect=['f' * 64, '/home/ale']), \
                 patch.object(sup, 'remote_scp', side_effect=lambda _h, _s, target: target.write_bytes(b'bad')):
                with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                    sup.snapshot_remote('sl-x86-1', '/var/lib/serverledge/profiling-samples.jsonl',
                                        destination, 'sl-x86-1_active.jsonl')
            self.assertFalse(destination.exists())

    def test_emergency_preserves_pending_and_raw(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'pending.json').write_text(json.dumps({'id':'r000_arm64_hashing', 'arch':'arm64'}))
            recorded = []
            def save(host, source, dest, basename):
                recorded.append((host, source))
                if host == 'sl-arm-1' and source.endswith('profiling-samples.jsonl'):
                    dest.write_bytes(b'{"schema_version":6}\n')
                    return hashlib.sha256(dest.read_bytes()).hexdigest()
                return 'ABSENT'
            with patch.object(sup, 'snapshot_remote', side_effect=save), \
                 patch.object(sup, 'remote_ssh', return_value='/home/ale'):
                self.assertTrue(sup.emergency_salvage(folder))
            rescue = next((folder / 'emergency').glob('rescue_*/rescue_manifest.json'))
            evidence = json.loads(rescue.read_text())
            self.assertEqual(evidence['pending_at_failure']['id'], 'r000_arm64_hashing')
            self.assertEqual(len(evidence['files']), 1)
            self.assertEqual(evidence['status'], 'SNAPSHOT_DI_EMERGENZA_NON_VALIDATO_RF')
            self.assertEqual(json.loads((folder / 'pending.json').read_text())['arch'], 'arm64')

    def test_delete_verifies_empty_resources(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            down = root / 'scripts/gcp_scripts/gcp_down.sh'
            down.parent.mkdir(parents=True)
            down.write_text('#!/bin/bash\n')
            calls = []
            def fake_run(argv, **kw):
                calls.append(argv)
                return SimpleNamespace(returncode=0)
            with patch.object(sup, 'ROOT', root), \
                 patch.object(sup, 'gc_list', return_value=[]) as list_fn, \
                 patch.object(sup.subprocess, 'run', side_effect=fake_run):
                self.assertTrue(sup.delete_cluster())
            self.assertEqual(calls, [['bash', str(down)]])
            self.assertEqual([call.args[0] for call in list_fn.call_args_list],
                             ['instances', 'instances', 'disks', 'addresses', 'snapshots'])

    def test_unknown_labeled_vm_refuses_delete(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            down = root / 'scripts/gcp_scripts/gcp_down.sh'
            down.parent.mkdir(parents=True)
            down.write_text('#!/bin/bash\n')
            with patch.object(sup, 'ROOT', root), \
                 patch.object(sup, 'gc_list', return_value=[{
                     'name':'unknown', 'zone':sup.ZONE,
                     'labels': {'progetto':'serverledge-tesi'}}]), \
                 patch.object(sup.subprocess, 'run') as run:
                self.assertFalse(sup.delete_cluster())
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
