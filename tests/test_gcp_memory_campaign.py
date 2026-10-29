"""Offline checks for the new GCP runner; never sends network calls."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.profiling import gcp_memory_campaign as app
from analysis.profiling.gcp_memory_prepare import CONFIGS
from analysis.profiling import gcp_memory_workload as client


@contextmanager
def fake_corpus():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        catalog=root/'scripts/gcp_scripts/functions_catalog.sh'
        catalog.parent.mkdir(parents=True)
        targets=[f'example-{i:02d}' for i in range(10)]
        catalog.write_text('FUNCTIONS=(\n'+''.join(f'"{t}|python314|{t}.py|1024|{t}.handler"\n' for t in targets)+')\n')
        plan=root/'data/profiling/gcp-memory/planning/target_selection_v1/target_selection.csv'
        plan.parent.mkdir(parents=True)
        with plan.open('w',newline='') as f:
            wr=csv.writer(f);wr.writerow(['function_name']);wr.writerows([[t] for t in targets])
        for t in targets:
            folder=root/'data/profiling/gcp-memory/priors'/f'{t}-r01'
            folder.mkdir(parents=True)
            conds=[]
            for cid,mode,wr,we,c in CONFIGS:
                item={'config_id':cid,'formula':mode,'policy':'UCB1Decoupled','c':c,
                      'prior_applied':mode!='none','w_R':wr,'w_E':we}
                if mode!='none':
                    b=folder/cid;b.mkdir()
                    prior={'policy':'UCB1Decoupled','donor_function_name':'donor-x',
                           'config':{'ucb1_reference_anchor':{'mode':mode}},
                           'arms':{arch:{'ucb1':{'observation_weight':wr,
                                                'exploration_observation_weight':we}}
                                   for arch in ('amd64','arm64')}}
                    text=json.dumps(prior)
                    (b/'frozen-prior.json').write_text(text)
                    digest=hashlib.sha256(text.encode()).hexdigest()
                    (b/'manifest.json').write_text(json.dumps({'prior_sha256':digest}))
                    (b/'materialized-request.json').write_text(json.dumps({
                        'target_function_name':t,'prior':prior}))
                    item['prior_sha256']=digest
                conds.append(item)
            (folder/'grid-manifest.json').write_text(json.dumps({'target':t,'replica':'r01',
                'donor':'donor-x','status':'OFFLINE_PREPARED_NOT_GCP_VALIDATED',
                'conditions':conds,'bootstrap_commit':app.PIN}))
        with patch.object(app,'ROOT',root),patch.object(app,'PLAN',plan),\
             patch.object(app,'PRIOR_ROOT',root/'data/profiling/gcp-memory/priors'):
            yield root,targets


class CampaignOffline(unittest.TestCase):
    def test_full_grid_preflight(self):
        with fake_corpus() as (_, targets),patch.object(app,'gc',side_effect=AssertionError('NO GCP')):
            result=app.preflight()
            self.assertEqual(result['targets'],targets)
            self.assertEqual(len(result['prior_sha256']),60)

    def test_tampered_prior_refused(self):
        with fake_corpus() as (root,targets):
            path=root/'data/profiling/gcp-memory/priors'/f'{targets[0]}-r01'/'R04'/'frozen-prior.json'
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(app.CampaignError,'SHA mismatch'):
                app.preflight()

    def test_duplicate_target_refused(self):
        with fake_corpus() as (root, targets):
            with app.PLAN.open('a') as f:f.write(f'{targets[0]}\n')
            with self.assertRaises(app.CampaignError):app.preflight()

    def test_partial_result_never_counts_as_verified(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'status.json').write_text('{"status":"STARTED"}')
            self.assertFalse(app.verified(p))

    def test_changed_instance_id_refused(self):
        doc={'vm_ids':{'sl-lb':'123'},'run_id':'mem20261024'}
        with patch.object(app,'instances',return_value={'sl-lb':{'id':'999',
            'labels':{'memrun':'mem20261024'},'status':'RUNNING'}}):
            with self.assertRaisesRegex(app.CampaignError,'changed GCP instance ID'):
                app.vm_guard(doc)

    def test_full_cleanup_requires_all_local_runs(self):
        with tempfile.TemporaryDirectory() as d:
            doc={'targets':['test'], 'configs':['N04'],'replicas':1,
                 'finished_grid_concurrency':3}
            with patch.object(app,'gc',side_effect=AssertionError('DELETE must not occur')):
                with self.assertRaisesRegex(app.CampaignError,'Missing/corrupt'):
                    app.finish(doc,Path(d),3)

    def test_missing_decision_detected(self):
        with self.assertRaisesRegex(app.CampaignError,'50 MAB'):
            app.check_decisions('event=decision_resolved function=test\n','test',50)

    def test_end_to_end_collector_with_simulated_worker_logs(self):
        """Exercise full local export/checkpoint path with fake SSH/SCP only."""
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)
            target='test-fn'
            logs={
                'sl-x86-1': '\n'.join([
                    '2026/10/24 [CONTAINER_LIFECYCLE] event=container_created ts=1000 container_id="a" function="test-fn" machine_tag="amd64" node="sl-x86-1" memory_mb=1024 creation_kind="cold_start"',
                    '2026/10/24 [CONTAINER_LIFECYCLE] event=container_created ts=1200 container_id="b" function="test-fn" machine_tag="amd64" node="sl-x86-1" memory_mb=1024 creation_kind="cold_start"',
                    '2026/10/24 [CONTAINER_LIFECYCLE] event=container_destroyed ts=1800 container_id="a" function="test-fn" machine_tag="amd64" node="sl-x86-1" memory_mb=1024 memory_lookup_ok=true reason="expired"',
                    '2026/10/24 [CONTAINER_LIFECYCLE] event=container_destroyed ts=2000 container_id="b" function="test-fn" machine_tag="amd64" node="sl-x86-1" memory_mb=1024 memory_lookup_ok=true reason="expired"',
                ]),
                'sl-arm-1': '\n'.join([
                    '2026/10/24 [CONTAINER_LIFECYCLE] event=container_created ts=1000 container_id="c" function="test-fn" machine_tag="arm64" node="sl-arm-1" memory_mb=1024 creation_kind="cold_start"',
                    '2026/10/24 [CONTAINER_LIFECYCLE] event=container_destroyed ts=1900 container_id="c" function="test-fn" machine_tag="arm64" node="sl-arm-1" memory_mb=1024 memory_lookup_ok=true reason="expired"',
                ])}
            lb='\n'.join(f'event=decision_resolved request_id=fake-{i} function={target} selected_arm=amd64 execution_arm=amd64 fallback=false fallback_reason=' for i in range(50))
            def fake_remote(host, script, **kwargs):
                return logs[host] if host in logs else ''
            def fake_download(host, source, output):
                output.parent.mkdir(parents=True,exist_ok=True)
                if output.name=='requests.jsonl':
                    payload='\n'.join(json.dumps({'index':i,'client_elapsed_ms':5,'success':True,
                            'is_warm_start':i%2==0, 'execution_header_arm':'amd64'}) for i in range(1,51))+'\n'
                else:payload='{}'
                output.write_text(payload)
            def fake_log(host,src,path):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(lb if host=='sl-lb' else logs[host])
            doc={'run_id':'mem20261024140000'}
            with patch.object(app,'fresh_runtime'),patch.object(app,'preflight',return_value={
                 'function_data':{target:{'catalog':{'runtime':'python314','src':'test.py',
                                                  'memory_mb':1024,'handler':'test.handler'}}}}),\
                 patch.object(app,'register_target'),patch.object(app,'assert_no_target_container_created'),patch.object(app,'apply_prior',return_value=None),\
                 patch.object(app,'upload'),patch.object(app,'internal_ip',return_value='10.0.0.1'),\
                 patch.object(app,'remote',side_effect=fake_remote),\
                 patch.object(app,'download',side_effect=fake_download),\
                 patch.object(app,'run_remote_file',side_effect=fake_log),\
                 patch.object(app,'ROOT',Path(__file__).resolve().parents[1]):
                result=app.collect_one(doc,dest,target,'N04',1,3,require_multi=True)
            self.assertEqual(result['created_containers'],3)
            self.assertEqual(result['max_containers_one_worker'],2)
            self.assertTrue(app.verified(dest/'runs'/'test-fn-N04-r01-q3'))

    def test_workload_count_and_concurrency_validation(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                client.execute('127.0.0.1','test',100,2,Path(d)/'calls.jsonl')


    def test_real_preflight_catalog_is_unwrapped_before_registration(self):
        # Regression for the real GCP failure KeyError: 'runtime':
        # preflight()['function_data'][name] contains nested 'catalog'.
        expected = {'runtime':'python312ml', 'src':'multilang/python/billibench_knn.py',
                    'memory_mb':1024, 'handler':'billibench_knn.handler'}
        with patch.object(app, 'preflight', return_value={
                 'function_data':{'billibench-knn':{'donor':'billibench-linear-regression',
                                                  'catalog':expected}}}):
            catalog = app.registration_catalog()
        self.assertEqual(catalog['billibench-knn'], expected)
        with patch.object(app, 'internal_ip', return_value='10.0.0.2') as ip, \
             patch.object(app, 'remote', return_value='') as ssh, \
             patch.object(app, 'assert_no_target_container_created') as no_prewarm:
            app.register_target('billibench-knn', catalog)
        ip.assert_called_once_with('sl-lb')
        ssh.assert_called_once()
        self.assertEqual(ssh.call_args[0][0], 'sl-workload')
        self.assertIn('--runtime python312ml', ssh.call_args[0][1])
        self.assertIn('--max_concurrency 1', ssh.call_args[0][1])
        no_prewarm.assert_called_once_with('billibench-knn', phase='function registration')

    def test_registration_rejects_missing_runtime_before_network(self):
        with patch.object(app, 'preflight', return_value={
                'function_data':{'target':{'catalog':{'src':'x.py'}}}}), \
             patch.object(app, 'gc', side_effect=AssertionError('NO GCP')):
            with self.assertRaisesRegex(app.CampaignError, 'incomplete catalog'):
                app.registration_catalog()

    def test_resume_requires_stopped_owned_instances(self):
        doc={'provisioned':True, 'deployed':True,
             'vm_ids':{name:f'id-{name}' for name in app.VMS}}
        terminated={name:{'status':'TERMINATED'} for name in app.VMS}
        running={name:{'status':'RUNNING'} for name in app.VMS}
        with patch.object(app,'vm_guard',side_effect=[terminated,running]), \
             patch.object(app,'gc',return_value='') as gc:
            app.resume_owned(doc)
        self.assertEqual(gc.call_count,1)
        self.assertEqual(gc.call_args.args[:3],('compute','instances','start'))
        for name in app.VMS:
            self.assertIn(name, gc.call_args.args)

    def test_resume_refuses_partial_running_or_changed_vm_ids(self):
        doc={'provisioned':True, 'deployed':True,
             'vm_ids':{name:f'id-{name}' for name in app.VMS}}
        mixed={name:{'status':'TERMINATED'} for name in app.VMS}
        mixed['sl-lb']={'status':'RUNNING'}
        with patch.object(app,'vm_guard',return_value=mixed), \
             patch.object(app,'gc',side_effect=AssertionError('must not start')):
            with self.assertRaisesRegex(app.CampaignError,'requires all five'):
                app.resume_owned(doc)



    def test_r04_one_realistic_fallback_is_a_valid_observation(self):
        """A realistic arm64->amd64 fallback must not discard a closed run."""
        name = 'billibench-knn'
        lines = []
        for i in range(50):
            fallback = i == 49
            lines.append(
                f'2026/10/25 07:05:13 [MAB] event=decision_resolved '
                f'request_id=req-{i} function={name} selected_arm=arm64 '
                f'execution_arm={"amd64" if fallback else "arm64"} '
                f'fallback={str(fallback).lower()} '
                f'fallback_reason={"selected_arm_no_candidate" if fallback else ""}'
            )
        lines.append(
            f'2026/10/25 07:05:13 [MAB] event=synthetic_reward '
            f'request_id=req-49 function={name} arm=arm64 '
            'reward=-20.000000 reason=selected_arm_no_candidate'
        )
        outcome = app.check_decisions('\n'.join(lines), name, 50)
        self.assertEqual(outcome['fallbacks'], 1)
        self.assertEqual(outcome['selected_arm_counts'], {'arm64': 50})
        self.assertEqual(outcome['execution_arm_counts'], {'arm64': 49, 'amd64': 1})
        self.assertEqual(outcome['fallback_reasons'], {'selected_arm_no_candidate': 1})
        self.assertTrue(outcome['fallback_events'][0]['synthetic_reward_logged'])
        self.assertEqual(outcome['comparison_class'], 'WITH_FALLBACK')

    def test_duplicate_mab_id_refused_even_if_count_is_50(self):
        lb = '\n'.join(
            'event=decision_resolved request_id=same function=target '
            'selected_arm=arm64 execution_arm=arm64 fallback=false fallback_reason='
            for _ in range(50)
        )
        with self.assertRaisesRegex(app.CampaignError, 'Duplicate/missing MAB'):
            app.check_decisions(lb, 'target', 50)

    def test_false_fallback_for_different_architectures_refused(self):
        lb = '\n'.join(
            f'event=decision_resolved request_id=id-{i} function=target '
            'selected_arm=arm64 execution_arm=amd64 fallback=false fallback_reason='
            for i in range(50)
        )
        with self.assertRaisesRegex(app.CampaignError, 'Inconsistent selected/execution'):
            app.check_decisions(lb, 'target', 50)

    def test_complete_fallback_run_can_be_recovered_offline(self):
        """Read saved raw artifacts; recover a run once without any GCP call."""
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)
            key = 'billibench-knn-R04-r01-q3'
            folder = dest / 'runs' / key
            (folder/'raw').mkdir(parents=True)
            target='billibench-knn'
            (folder/'status.json').write_text(json.dumps({
                'key':key,'target':target,'condition':'R04','replica':1,
                'concurrency':3,'requests':50,'status':'STARTED','runtime_pin':app.PIN,
            }))
            worker_arch = {'sl-x86-1': 'amd64', 'sl-arm-1': 'arm64'}
            for worker, arch in worker_arch.items():
                cid = 'x' if arch=='amd64' else 'a'
                (folder/'raw'/f'{worker}.log').write_text('\n'.join([
                    f'[CONTAINER_LIFECYCLE] event=container_created ts=1000 container_id="{cid}" function="{target}" machine_tag="{arch}" node="{worker}" memory_mb=1024 creation_kind="cold_start"',
                    f'[CONTAINER_LIFECYCLE] event=container_destroyed ts=2000 container_id="{cid}" function="{target}" machine_tag="{arch}" node="{worker}" memory_mb=1024 memory_lookup_ok=true reason="expired"',
                ]))
            lines = [
                f'event=decision_resolved request_id=id-{i} function={target} '
                f'selected_arm=arm64 execution_arm={"amd64" if i==49 else "arm64"} '
                f'fallback={str(i==49).lower()} '
                f'fallback_reason={"selected_arm_no_candidate" if i==49 else ""}'
                for i in range(50)
            ]
            (folder/'raw/sl-lb.log').write_text('\n'.join(lines))
            (folder/'requests.jsonl').write_text('\n'.join(json.dumps({
                'index':i+1,'success':True,'client_elapsed_ms':20.0,
                'is_warm_start':True,'execution_header_arm': 'amd64' if i==49 else 'arm64'
            }) for i in range(50)) + '\n')
            src_prior = dest / 'priors'/f'{target}-r01'/'R04'/'materialized-request.json'
            src_prior.parent.mkdir(parents=True)
            src_prior.write_text(json.dumps({'target_function_name':target,'prior':{'x':1}}))
            (folder/'transfer-response.json').write_text(json.dumps({
                'transfer_attempted':True,'transfer_applied':True,
                'initialization_source':'materialized_prior',
                'target_function_name':target,'prior':{'x':1}
            }))
            doc = {'targets':[target], 'configs':['R04'], 'replicas':1}
            with patch.object(app,'PRIOR_ROOT',dest/'priors'), \
                 patch.object(app,'gc',side_effect=AssertionError('NO GCP')):
                audit = app.inspect_existing_run(doc,dest,key,recover=False)
                self.assertEqual(audit['mab']['fallbacks'],1)
                self.assertFalse((folder/'verified.json').exists())
                out = app.inspect_existing_run(doc,dest,key,recover=True)
                self.assertEqual(out['validation_class'],'COMPLETE_WITH_FALLBACK')
                self.assertTrue(app.verified(folder))
                self.assertEqual(app.load(folder/'status.json')['status'],'VERIFIED_LOCAL')
                with self.assertRaisesRegex(app.CampaignError, 'already finalized'):
                    app.inspect_existing_run(doc,dest,key,recover=True)
                self.assertTrue((folder/'local-recovery-provenance.json').exists())
                self.assertEqual(app.load(folder/'verified.json')['fallbacks'],1)

    def test_malformed_headers_block_offline_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            (folder/'requests.jsonl').write_text('\n'.join(json.dumps({
                'index':i,'success':True,'client_elapsed_ms':5,
                'execution_header_arm':''
            }) for i in range(1,51)))
            (folder/'raw').mkdir()
            (folder/'raw/sl-lb.log').write_text('')
            for host in app.WORKERS:
                (folder/'raw'/f'{host}.log').write_text('')
            with self.assertRaisesRegex(app.CampaignError,'response architecture header'):
                app.validate_local_measurement(folder,'test','N04',3)



    def test_batched_full_grid_resume_and_complete(self):
        """Three separate sessions advance checkpoints without redoing work."""
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)
            doc={'mode':'full','deployed':True,'targets':['test-fn'],
                 'configs':['N04','D04','R04'],'replicas':1}
            (dest/'runs').mkdir()
            completed=set()
            calls=[]
            def fake_collect(document, out, target, cid, replica, concurrency):
                key=f'{target}-{cid}-r{replica:02d}-q{concurrency}'
                calls.append(key)
                (out/'runs'/key).mkdir()
                completed.add(key)
            with patch.object(app,'collect_one',side_effect=fake_collect), \
                 patch.object(app,'verified',side_effect=lambda folder: folder.name in completed):
                self.assertFalse(app.run_grid(doc,dest,concurrency=3,max_new_runs=1))
                self.assertEqual(len(calls),1)
                self.assertEqual(doc['run_concurrency'],3)
                self.assertNotIn('finished_grid_concurrency',doc)
                self.assertFalse(app.run_grid(doc,dest,concurrency=3,max_new_runs=1))
                self.assertEqual(len(calls),2)
                self.assertTrue(app.run_grid(doc,dest,concurrency=3,max_new_runs=1))
                self.assertEqual(len(calls),3)
                self.assertEqual(doc['finished_grid_concurrency'],3)
                self.assertTrue(app.run_grid(doc,dest,concurrency=3,max_new_runs=1))
                self.assertEqual(len(calls),3)
            saved=app.load(dest/'campaign.json')
            self.assertEqual(len(saved['batch_history']),4)
            self.assertEqual(saved['batch_history'][-1]['cumulative_verified'],3)

    def test_full_campaign_refuses_unbounded_and_concurrency_change(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)
            (dest/'runs').mkdir()
            doc={'mode':'full','deployed':True,'targets':['fn'],
                 'configs':['N04'],'replicas':1}
            with patch.object(app,'collect_one',side_effect=AssertionError('no cloud')):
                with self.assertRaisesRegex(app.CampaignError,'requires --max-runs'):
                    app.run_grid(doc,dest,concurrency=3)
            doc['run_concurrency']=3
            with self.assertRaisesRegex(app.CampaignError,'locked to concurrency 3'):
                app.run_grid(doc,dest,concurrency=6,max_new_runs=1)

    def test_batch_refuses_corrupt_existing_measurement(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)
            partial=dest/'runs'/'fn-N04-r01-q3'
            partial.mkdir(parents=True)
            doc={'mode':'full','deployed':True,'targets':['fn'],
                 'configs':['N04'],'replicas':1}
            with patch.object(app,'collect_one',side_effect=AssertionError('must not rerun')):
                with self.assertRaisesRegex(app.CampaignError,'incomplete/corrupted'):
                    app.run_grid(doc,dest,concurrency=3,max_new_runs=1)

    def test_batch_size_limit(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)
            (dest/'runs').mkdir()
            doc={'mode':'full','deployed':True,'targets':['fn'],
                 'configs':['N04'],'replicas':1}
            for limit in (0,21,-1):
                with self.assertRaisesRegex(app.CampaignError,'between 1 and 20'):
                    app.run_grid(doc,dest,concurrency=3,max_new_runs=limit)

if __name__=='__main__': unittest.main()
