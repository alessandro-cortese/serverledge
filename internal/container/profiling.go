package container

import (
	"context"
	"fmt"
	"time"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/serverledge-faas/serverledge/internal/executor"
	"github.com/serverledge-faas/serverledge/internal/profiling"
)

const (
	defaultKeplerMetricsURL = "http://127.0.0.1:28282/metrics"

	defaultKeplerHTTPTimeout = 3 * time.Second

	defaultKeplerPollInterval = 100 * time.Millisecond

	defaultKeplerRefreshTimeout = 3 * time.Second
)

type keplerCollectionConfig struct {
	client *profiling.KeplerClient

	pollInterval   time.Duration
	refreshTimeout time.Duration
}

func configuredKeplerCollection() (
	*keplerCollectionConfig,
	error,
) {
	httpTimeoutMs :=
		config.GetInt(
			config.FUNCTION_PROFILING_KEPLER_TIMEOUT_MS,
			int(
				defaultKeplerHTTPTimeout/
					time.Millisecond,
			),
		)

	pollIntervalMs :=
		config.GetInt(
			config.FUNCTION_PROFILING_KEPLER_POLL_INTERVAL_MS,
			int(
				defaultKeplerPollInterval/
					time.Millisecond,
			),
		)

	refreshTimeoutMs :=
		config.GetInt(
			config.FUNCTION_PROFILING_KEPLER_REFRESH_TIMEOUT_MS,
			int(
				defaultKeplerRefreshTimeout/
					time.Millisecond,
			),
		)

	if httpTimeoutMs <= 0 {
		return nil,
			fmt.Errorf(
				"profiling.kepler.timeout_ms must be positive",
			)
	}

	if pollIntervalMs <= 0 {
		return nil,
			fmt.Errorf(
				"profiling.kepler.poll_interval_ms must be positive",
			)
	}

	if refreshTimeoutMs <= 0 {
		return nil,
			fmt.Errorf(
				"profiling.kepler.refresh_timeout_ms must be positive",
			)
	}

	client,
		err :=
		profiling.NewKeplerClient(
			config.GetString(
				config.FUNCTION_PROFILING_KEPLER_URL,
				defaultKeplerMetricsURL,
			),
			time.Duration(
				httpTimeoutMs,
			)*time.Millisecond,
		)

	if err != nil {
		return nil,
			err
	}

	return &keplerCollectionConfig{
			client: client,

			pollInterval: time.Duration(
				pollIntervalMs,
			) * time.Millisecond,

			refreshTimeout: time.Duration(
				refreshTimeoutMs,
			) * time.Millisecond,
		},
		nil
}

// ExecuteProfiled invokes one function.
//
// Docker/node resource profiling and Kepler energy collection are deliberately
// separate concerns:
//
//   - resource profiling is controlled by profiling.enabled together with
//     collectResourceProfile and is used by the profiling/clustering pipeline;
//   - Kepler collection is controlled independently through
//     profiling.kepler.enabled and produces an execution-level energy
//     measurement.
//
// When either measurement requires an exclusive window, profilingMu remains
// locked until all final measurements, including a refresh-aware Kepler read,
// have completed.
func ExecuteProfiled(
	cont *Container,
	req *executor.InvocationRequest,
	maxConcurrency int16,
	collectResourceProfile bool,
) (
	*executor.InvocationResult,
	time.Duration,
	time.Duration,
	*profiling.InvocationResourceProfile,
	*profiling.NodeResourceProfile,
	*profiling.KeplerInvocationEnergyProfile,
	error,
) {
	if cont == nil {
		return nil,
			0,
			0,
			nil,
			nil,
			nil,
			fmt.Errorf(
				"cannot execute on a nil container",
			)
	}

	contID :=
		cont.ID

	resourceProfilingEnabled :=
		config.GetBool(
			config.FUNCTION_PROFILING_ENABLED,
			false,
		) &&
			collectResourceProfile

	keplerEnabled :=
		config.GetBool(
			config.FUNCTION_PROFILING_KEPLER_ENABLED,
			false,
		)

	if !resourceProfilingEnabled &&
		!keplerEnabled {

		executionStartedAt :=
			time.Now()

		response,
			invocationWait,
			err :=
			Execute(
				contID,
				req,
			)

		executionWallTime :=
			time.Since(
				executionStartedAt,
			)

		return response,
			invocationWait,
			executionWallTime,
			nil,
			nil,
			nil,
			err
	}

	profilingLockStartedAt :=
		time.Now()

	cont.profilingMu.Lock()

	profilingLockWait :=
		time.Since(
			profilingLockStartedAt,
		)

	defer cont.profilingMu.Unlock()

	var nodeBefore profiling.NodeResourceSnapshot
	var nodeBeforeErr error
	var nodeStartOverhead time.Duration

	var containerBefore profiling.ResourceSnapshot
	var containerBeforeErr error
	var containerStartOverhead time.Duration

	if resourceProfilingEnabled {
		// The node snapshot is intentionally taken first so that the node
		// profiling interval contains the container profiling interval.
		nodeStartSnapshotStartedAt :=
			time.Now()

		nodeBefore,
			nodeBeforeErr =
			profiling.ReadNodeResourceSnapshot()

		nodeStartOverhead =
			time.Since(
				nodeStartSnapshotStartedAt,
			)

		containerStartSnapshotStartedAt :=
			time.Now()

		containerBefore,
			containerBeforeErr =
			cf.GetResourceSnapshot(
				contID,
			)

		containerStartOverhead =
			time.Since(
				containerStartSnapshotStartedAt,
			)
	}

	var keplerConfig *keplerCollectionConfig
	var keplerBefore profiling.KeplerContainerSnapshot
	var keplerBeforeAvailable bool
	var keplerEnergyProfile *profiling.KeplerInvocationEnergyProfile

	if keplerEnabled {
		var keplerConfigErr error

		keplerConfig,
			keplerConfigErr =
			configuredKeplerCollection()

		if keplerConfigErr != nil {
			keplerEnergyProfile =
				profiling.NewInvalidKeplerInvocationEnergyProfile(
					contID,
					fmt.Sprintf(
						"kepler_configuration_failed: %v",
						keplerConfigErr,
					),
				)
		} else {
			keplerBefore,
				keplerConfigErr =
				keplerConfig.client.
					ReadContainerSnapshot(
						context.Background(),
						contID,
					)

			if keplerConfigErr != nil {
				keplerEnergyProfile =
					profiling.NewInvalidKeplerInvocationEnergyProfile(
						contID,
						fmt.Sprintf(
							"kepler_snapshot_before_failed: %v",
							keplerConfigErr,
						),
					)
			} else {
				keplerBeforeAvailable =
					true
			}
		}
	}

	executionStartedAt :=
		time.Now()

	response,
		invocationWait,
		executionErr :=
		Execute(
			contID,
			req,
		)

	executionWallTime :=
		time.Since(
			executionStartedAt,
		)

	var containerAfter profiling.ResourceSnapshot
	var containerAfterErr error
	var containerEndOverhead time.Duration

	var nodeAfter profiling.NodeResourceSnapshot
	var nodeAfterErr error
	var nodeEndOverhead time.Duration

	if resourceProfilingEnabled {
		if containerBeforeErr == nil {
			containerEndSnapshotStartedAt :=
				time.Now()

			containerAfter,
				containerAfterErr =
				cf.GetResourceSnapshot(
					contID,
				)

			containerEndOverhead =
				time.Since(
					containerEndSnapshotStartedAt,
				)
		}

		nodeEndSnapshotStartedAt :=
			time.Now()

		nodeAfter,
			nodeAfterErr =
			profiling.ReadNodeResourceSnapshot()

		nodeEndOverhead =
			time.Since(
				nodeEndSnapshotStartedAt,
			)
	}

	// Kepler is deliberately read only after the execution and after the
	// ordinary final snapshots. profilingMu is still held here, therefore no
	// second invocation handled through ExecuteProfiled can start on this
	// container while Serverledge waits for Kepler to refresh its counters.
	if keplerEnabled &&
		keplerBeforeAvailable {

		refreshContext,
			cancel :=
			context.WithTimeout(
				context.Background(),
				keplerConfig.refreshTimeout,
			)

		var keplerAfterErr error

		keplerEnergyProfile,
			keplerAfterErr =
			keplerConfig.client.
				WaitForContainerEnergyDelta(
					refreshContext,
					keplerBefore,
					keplerConfig.pollInterval,
				)

		cancel()

		if keplerAfterErr != nil {
			keplerEnergyProfile =
				profiling.NewInvalidKeplerInvocationEnergyProfile(
					contID,
					fmt.Sprintf(
						"kepler_snapshot_after_failed: %v",
						keplerAfterErr,
					),
				)
		}
	}

	var resourceProfile *profiling.InvocationResourceProfile

	if resourceProfilingEnabled {
		switch {
		case containerBeforeErr != nil:
			resourceProfile =
				profiling.NewInvalidInvocationResourceProfile(
					contID,
					maxConcurrency,
					true,
					profilingLockWait,
					fmt.Sprintf(
						"snapshot_before_failed: %v",
						containerBeforeErr,
					),
					containerStartOverhead,
					containerEndOverhead,
				)

		case containerAfterErr != nil:
			resourceProfile =
				profiling.NewInvalidInvocationResourceProfile(
					contID,
					maxConcurrency,
					true,
					profilingLockWait,
					fmt.Sprintf(
						"snapshot_after_failed: %v",
						containerAfterErr,
					),
					containerStartOverhead,
					containerEndOverhead,
				)

		default:
			resourceProfile =
				profiling.BuildInvocationResourceProfile(
					contID,
					maxConcurrency,
					true,
					profilingLockWait,
					containerBefore,
					containerAfter,
					executionWallTime,
					containerStartOverhead,
					containerEndOverhead,
				)
		}
	}

	var nodeProfile *profiling.NodeResourceProfile

	if resourceProfilingEnabled {
		switch {
		case nodeBeforeErr != nil:
			nodeProfile =
				profiling.NewInvalidNodeResourceProfile(
					fmt.Sprintf(
						"node_snapshot_before_failed: %v",
						nodeBeforeErr,
					),
					nodeStartOverhead,
					nodeEndOverhead,
				)

		case nodeAfterErr != nil:
			nodeProfile =
				profiling.NewInvalidNodeResourceProfile(
					fmt.Sprintf(
						"node_snapshot_after_failed: %v",
						nodeAfterErr,
					),
					nodeStartOverhead,
					nodeEndOverhead,
				)

		default:
			nodeProfile =
				profiling.BuildNodeResourceProfile(
					nodeBefore,
					nodeAfter,
					executionWallTime,
					nodeStartOverhead,
					nodeEndOverhead,
				)
		}
	}

	return response,
		invocationWait,
		executionWallTime,
		resourceProfile,
		nodeProfile,
		keplerEnergyProfile,
		executionErr
}
