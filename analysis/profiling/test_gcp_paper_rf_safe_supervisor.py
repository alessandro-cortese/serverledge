#!/usr/bin/env python3
"""Offline regression tests: no calls to GCP."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gcp_paper_rf_supervisor as s


def item(name, status):
    return {'name': name, 'status': status, 'zone': 'https://example/zones/europe-west4-a'}


class SafeSupervisorTests(unittest.TestCase):
    def test_stop_preserves_disks_and_targets_only_expected_vms(self):
        before = [item(n, 'RUNNING' if n != 'sl-registry' else 'TERMINATED') for n in s.VM_NAMES]
        after = [item(n, 'TERMINATED') for n in s.VM_NAMES]
        before.append(item('someone-elses-vm', 'RUNNING'))
        after.append(item('someone-elses-vm', 'RUNNING'))
        with mock.patch.object(s, 'gc_list', side_effect=[before, after]) as ls, \
             mock.patch.object(s.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stderr='')) as run:
            self.assertTrue(s.stop_cluster_preserving_disks())
            args = run.call_args.args[0]
            self.assertEqual(args[:4], ['gcloud','compute','instances','stop'])
            self.assertNotIn('sl-registry', args)
            self.assertNotIn('someone-elses-vm', args)
            self.assertNotIn('delete', args)
            self.assertEqual(ls.call_count, 2)

    def test_stop_no_network_reports_failure_not_success(self):
        with mock.patch.object(s, 'gc_list', side_effect=RuntimeError('DNS failed')):
            self.assertFalse(s.stop_cluster_preserving_disks())

    def test_stop_missing_vm_is_not_treated_as_success(self):
        with mock.patch.object(s, 'gc_list', return_value=[item(n, 'TERMINATED') for n in s.VM_NAMES[:-1]]):
            self.assertFalse(s.stop_cluster_preserving_disks())

    def test_failure_preserve_on_error_does_not_delete(self):
        with tempfile.TemporaryDirectory() as td, \
            mock.patch.object(s, 'validate_launch_manifest', return_value={'ok': True}), \
            mock.patch.object(s, 'supervise', side_effect=RuntimeError('network lost')), \
            mock.patch.object(s, 'try_recover_checkpoint'), \
            mock.patch.object(s, 'emergency_salvage', return_value=False), \
            mock.patch.object(s, 'stop_cluster_preserving_disks', return_value=True) as stop, \
            mock.patch.object(s, 'delete_cluster') as delete:
            result = s.main(['--dest',td,'--run','--confirm-delete','--preserve-on-error'])
            self.assertEqual(result, 1)
            stop.assert_called_once_with()
            delete.assert_not_called()

    def test_success_runs_delete_even_with_preserve_on_error(self):
        with tempfile.TemporaryDirectory() as td, \
            mock.patch.object(s, 'validate_launch_manifest', return_value={'ok': True}), \
            mock.patch.object(s, 'supervise', return_value=None), \
            mock.patch.object(s, 'stop_cluster_preserving_disks') as stop, \
            mock.patch.object(s, 'delete_cluster', return_value=True) as delete:
            result = s.main(['--dest',td,'--run','--confirm-delete','--preserve-on-error'])
            self.assertEqual(result, 0)
            delete.assert_called_once_with()
            stop.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
