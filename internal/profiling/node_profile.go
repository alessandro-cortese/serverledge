package profiling

import (
	"bufio"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const defaultProcRoot = "/proc"

// NodeCPUStatSnapshot contains cumulative host/VM CPU counters read from
// /proc/stat. Values are expressed in Linux clock ticks.
type NodeCPUStatSnapshot struct {
	UserTicks      uint64
	NiceTicks      uint64
	KernelTicks    uint64
	IdleTicks      uint64
	IOWaitTicks    uint64
	IRQTicks       uint64
	SoftIRQTicks   uint64
	StealTicks     uint64
	GuestTicks     uint64
	GuestNiceTicks uint64
	AvailableCPUs  int
}

// NodeMemorySnapshot contains point-in-time host/VM memory values read from
// /proc/meminfo.
type NodeMemorySnapshot struct {
	TotalBytes     uint64
	FreeBytes      uint64
	AvailableBytes uint64
}

// NodeVMStatSnapshot contains cumulative host/VM counters read from
// /proc/vmstat.
type NodeVMStatSnapshot struct {
	PageFaults      uint64
	MajorPageFaults uint64
}

// NodeResourceSnapshot is a point-in-time view of node-scoped counters.
//
// These values describe the operating-system environment visible to
// Serverledge and are intentionally kept separate from container-scoped
// Docker/cgroup data.
type NodeResourceSnapshot struct {
	ReadAt time.Time

	CPU    NodeCPUStatSnapshot
	Memory NodeMemorySnapshot
	VMStat NodeVMStatSnapshot

	CPUAvailable     bool
	MemoryAvailable  bool
	VMStatAvailable  bool
	CollectionErrors []string
}

// NodeResourceProfile contains node-scoped measurements for the same interval
// in which one warm invocation is profiled.
//
// These values describe the node or VM environment. They must not be
// interpreted as exclusive consumption by the function.
type NodeResourceProfile struct {
	Collected bool
	Complete  bool
	Errors    []string `json:",omitempty"`

	CPUAvailable    bool
	MemoryAvailable bool
	VMStatAvailable bool

	AvailableCPUs       int
	ExecutionWallTimeMs float64

	CPUUserDeltaTicks      uint64
	CPUNiceDeltaTicks      uint64
	CPUKernelDeltaTicks    uint64
	CPUIdleDeltaTicks      uint64
	CPUIOWaitDeltaTicks    uint64
	CPUIRQDeltaTicks       uint64
	CPUSoftIRQDeltaTicks   uint64
	CPUStealDeltaTicks     uint64
	CPUGuestDeltaTicks     uint64
	CPUGuestNiceDeltaTicks uint64
	CPUTotalDeltaTicks     uint64

	CPUUserDeltaMs      float64
	CPUNiceDeltaMs      float64
	CPUKernelDeltaMs    float64
	CPUIdleDeltaMs      float64
	CPUIOWaitDeltaMs    float64
	CPUIRQDeltaMs       float64
	CPUSoftIRQDeltaMs   float64
	CPUStealDeltaMs     float64
	CPUGuestDeltaMs     float64
	CPUGuestNiceDeltaMs float64

	CPUUserPercent      float64
	CPUNicePercent      float64
	CPUKernelPercent    float64
	CPUIdlePercent      float64
	CPUIOWaitPercent    float64
	CPUIRQPercent       float64
	CPUSoftIRQPercent   float64
	CPUStealPercent     float64
	CPUGuestPercent     float64
	CPUGuestNicePercent float64

	TotalMemoryBeforeBytes uint64
	TotalMemoryAfterBytes  uint64

	FreeMemoryBeforeBytes  uint64
	FreeMemoryAfterBytes   uint64
	FreeMemoryAverageBytes uint64

	AvailableMemoryBeforeBytes  uint64
	AvailableMemoryAfterBytes   uint64
	AvailableMemoryAverageBytes uint64

	PageFaultsBefore uint64
	PageFaultsAfter  uint64

	MajorPageFaultsBefore uint64
	MajorPageFaultsAfter  uint64

	PageFaultsDelta      uint64
	MajorPageFaultsDelta uint64

	SnapshotStartOverheadMs float64
	SnapshotEndOverheadMs   float64
	SnapshotTotalOverheadMs float64
}

// ReadNodeResourceSnapshot reads all supported node-scoped procfs sources.
//
// Partial collection is accepted. The returned snapshot records availability
// and errors independently for CPU, memory and VM statistics.
//
// An error is returned only when none of the three sources can be collected.
func ReadNodeResourceSnapshot() (NodeResourceSnapshot, error) {
	return readNodeResourceSnapshotFromRoot(defaultProcRoot)
}

func readNodeResourceSnapshotFromRoot(
	procRoot string,
) (NodeResourceSnapshot, error) {
	snapshot := NodeResourceSnapshot{
		ReadAt: time.Now(),
	}

	var collectionErrors []error

	statData, err :=
		os.ReadFile(
			filepath.Join(
				procRoot,
				"stat",
			),
		)

	if err != nil {
		message :=
			fmt.Sprintf(
				"proc_stat_read_failed: %v",
				err,
			)

		snapshot.CollectionErrors =
			append(
				snapshot.CollectionErrors,
				message,
			)

		collectionErrors =
			append(
				collectionErrors,
				errors.New(message),
			)
	} else {
		cpu, parseErr :=
			parseProcStat(
				statData,
			)

		if parseErr != nil {
			message :=
				fmt.Sprintf(
					"proc_stat_parse_failed: %v",
					parseErr,
				)

			snapshot.CollectionErrors =
				append(
					snapshot.CollectionErrors,
					message,
				)

			collectionErrors =
				append(
					collectionErrors,
					errors.New(message),
				)
		} else {
			snapshot.CPU = cpu
			snapshot.CPUAvailable = true
		}
	}

	memInfoData, err :=
		os.ReadFile(
			filepath.Join(
				procRoot,
				"meminfo",
			),
		)

	if err != nil {
		message :=
			fmt.Sprintf(
				"proc_meminfo_read_failed: %v",
				err,
			)

		snapshot.CollectionErrors =
			append(
				snapshot.CollectionErrors,
				message,
			)

		collectionErrors =
			append(
				collectionErrors,
				errors.New(message),
			)
	} else {
		memory, parseErr :=
			parseProcMemInfo(
				memInfoData,
			)

		if parseErr != nil {
			message :=
				fmt.Sprintf(
					"proc_meminfo_parse_failed: %v",
					parseErr,
				)

			snapshot.CollectionErrors =
				append(
					snapshot.CollectionErrors,
					message,
				)

			collectionErrors =
				append(
					collectionErrors,
					errors.New(message),
				)
		} else {
			snapshot.Memory = memory
			snapshot.MemoryAvailable = true
		}
	}

	vmStatData, err :=
		os.ReadFile(
			filepath.Join(
				procRoot,
				"vmstat",
			),
		)

	if err != nil {
		message :=
			fmt.Sprintf(
				"proc_vmstat_read_failed: %v",
				err,
			)

		snapshot.CollectionErrors =
			append(
				snapshot.CollectionErrors,
				message,
			)

		collectionErrors =
			append(
				collectionErrors,
				errors.New(message),
			)
	} else {
		vmStat, parseErr :=
			parseProcVMStat(
				vmStatData,
			)

		if parseErr != nil {
			message :=
				fmt.Sprintf(
					"proc_vmstat_parse_failed: %v",
					parseErr,
				)

			snapshot.CollectionErrors =
				append(
					snapshot.CollectionErrors,
					message,
				)

			collectionErrors =
				append(
					collectionErrors,
					errors.New(message),
				)
		} else {
			snapshot.VMStat = vmStat
			snapshot.VMStatAvailable = true
		}
	}

	if !snapshot.CPUAvailable &&
		!snapshot.MemoryAvailable &&
		!snapshot.VMStatAvailable {

		return snapshot,
			errors.Join(
				collectionErrors...,
			)
	}

	return snapshot, nil
}

// NewInvalidNodeResourceProfile creates a profile describing a complete
// failure of the initial or final node snapshot.
//
// The failure remains observability data and never prevents the invocation.
func NewInvalidNodeResourceProfile(
	reason string,
	startOverhead time.Duration,
	endOverhead time.Duration,
) *NodeResourceProfile {
	return &NodeResourceProfile{
		Collected: false,
		Complete:  false,

		Errors: []string{
			reason,
		},

		SnapshotStartOverheadMs: durationMilliseconds(
			startOverhead,
		),

		SnapshotEndOverheadMs: durationMilliseconds(
			endOverhead,
		),

		SnapshotTotalOverheadMs: durationMilliseconds(
			startOverhead +
				endOverhead,
		),
	}
}

// BuildNodeResourceProfile calculates node-scoped values and deltas from two
// procfs snapshots.
//
// The raw CPU tick deltas are retained for auditability. CPU milliseconds and
// percentages are normalized over the CPU time theoretically available during
// the measured wall-clock interval.
func BuildNodeResourceProfile(
	before NodeResourceSnapshot,
	after NodeResourceSnapshot,
	executionWallTime time.Duration,
	startOverhead time.Duration,
	endOverhead time.Duration,
) *NodeResourceProfile {
	profile :=
		&NodeResourceProfile{
			Collected: before.CPUAvailable ||
				before.MemoryAvailable ||
				before.VMStatAvailable ||
				after.CPUAvailable ||
				after.MemoryAvailable ||
				after.VMStatAvailable,

			CPUAvailable: before.CPUAvailable &&
				after.CPUAvailable,

			MemoryAvailable: before.MemoryAvailable &&
				after.MemoryAvailable,

			VMStatAvailable: before.VMStatAvailable &&
				after.VMStatAvailable,

			ExecutionWallTimeMs: durationMilliseconds(
				executionWallTime,
			),

			SnapshotStartOverheadMs: durationMilliseconds(
				startOverhead,
			),

			SnapshotEndOverheadMs: durationMilliseconds(
				endOverhead,
			),

			SnapshotTotalOverheadMs: durationMilliseconds(
				startOverhead +
					endOverhead,
			),
		}

	profile.Errors =
		append(
			profile.Errors,
			prefixNodeErrors(
				"before",
				before.CollectionErrors,
			)...,
		)

	profile.Errors =
		append(
			profile.Errors,
			prefixNodeErrors(
				"after",
				after.CollectionErrors,
			)...,
		)

	if profile.CPUAvailable {
		err :=
			populateNodeCPUProfile(
				profile,
				before.CPU,
				after.CPU,
			)

		if err != nil {
			profile.CPUAvailable = false

			profile.Errors =
				append(
					profile.Errors,
					err.Error(),
				)
		} else {
			populateNormalizedNodeCPUTimes(
				profile,
			)
		}
	} else {
		profile.Errors =
			append(
				profile.Errors,
				"cpu_snapshot_incomplete",
			)
	}

	if profile.MemoryAvailable {
		profile.TotalMemoryBeforeBytes =
			before.Memory.TotalBytes

		profile.TotalMemoryAfterBytes =
			after.Memory.TotalBytes

		profile.FreeMemoryBeforeBytes =
			before.Memory.FreeBytes

		profile.FreeMemoryAfterBytes =
			after.Memory.FreeBytes

		profile.FreeMemoryAverageBytes =
			averageNodeUint64(
				before.Memory.FreeBytes,
				after.Memory.FreeBytes,
			)

		profile.AvailableMemoryBeforeBytes =
			before.Memory.AvailableBytes

		profile.AvailableMemoryAfterBytes =
			after.Memory.AvailableBytes

		profile.AvailableMemoryAverageBytes =
			averageNodeUint64(
				before.Memory.AvailableBytes,
				after.Memory.AvailableBytes,
			)
	} else {
		profile.Errors =
			append(
				profile.Errors,
				"memory_snapshot_incomplete",
			)
	}

	if profile.VMStatAvailable {
		profile.PageFaultsBefore =
			before.VMStat.PageFaults

		profile.PageFaultsAfter =
			after.VMStat.PageFaults

		profile.MajorPageFaultsBefore =
			before.VMStat.MajorPageFaults

		profile.MajorPageFaultsAfter =
			after.VMStat.MajorPageFaults

		var ok bool

		profile.PageFaultsDelta, ok =
			counterDelta(
				after.VMStat.PageFaults,
				before.VMStat.PageFaults,
			)

		if !ok {
			profile.VMStatAvailable = false

			profile.Errors =
				append(
					profile.Errors,
					"node_page_fault_counter_regressed",
				)
		}

		profile.MajorPageFaultsDelta, ok =
			counterDelta(
				after.VMStat.MajorPageFaults,
				before.VMStat.MajorPageFaults,
			)

		if !ok {
			profile.VMStatAvailable = false

			profile.Errors =
				append(
					profile.Errors,
					"node_major_page_fault_counter_regressed",
				)
		}
	} else {
		profile.Errors =
			append(
				profile.Errors,
				"vmstat_snapshot_incomplete",
			)
	}

	profile.Complete =
		profile.CPUAvailable &&
			profile.MemoryAvailable &&
			profile.VMStatAvailable

	return profile
}

func parseProcStat(
	data []byte,
) (NodeCPUStatSnapshot, error) {
	var result NodeCPUStatSnapshot
	var aggregateFound bool

	scanner :=
		bufio.NewScanner(
			strings.NewReader(
				string(data),
			),
		)

	for scanner.Scan() {
		fields :=
			strings.Fields(
				scanner.Text(),
			)

		if len(fields) == 0 {
			continue
		}

		if fields[0] == "cpu" {
			if len(fields) < 5 {
				return result,
					fmt.Errorf(
						"aggregate cpu line has %d fields",
						len(fields),
					)
			}

			values :=
				make(
					[]uint64,
					10,
				)

			for index := 1; index < len(fields) &&
				index <= 10; index++ {

				value, err :=
					strconv.ParseUint(
						fields[index],
						10,
						64,
					)

				if err != nil {
					return result,
						fmt.Errorf(
							"invalid cpu field %d (%q): %w",
							index,
							fields[index],
							err,
						)
				}

				values[index-1] =
					value
			}

			result.UserTicks = values[0]
			result.NiceTicks = values[1]
			result.KernelTicks = values[2]
			result.IdleTicks = values[3]
			result.IOWaitTicks = values[4]
			result.IRQTicks = values[5]
			result.SoftIRQTicks = values[6]
			result.StealTicks = values[7]
			result.GuestTicks = values[8]
			result.GuestNiceTicks = values[9]

			aggregateFound = true

			continue
		}

		if isPerCPUName(
			fields[0],
		) {
			result.AvailableCPUs++
		}
	}

	if err := scanner.Err(); err != nil {
		return result,
			fmt.Errorf(
				"scan /proc/stat: %w",
				err,
			)
	}

	if !aggregateFound {
		return result,
			errors.New(
				"aggregate cpu line not found",
			)
	}

	if result.AvailableCPUs == 0 {
		return result,
			errors.New(
				"no per-cpu lines found",
			)
	}

	return result, nil
}

func parseProcMemInfo(
	data []byte,
) (NodeMemorySnapshot, error) {
	values :=
		make(
			map[string]uint64,
		)

	scanner :=
		bufio.NewScanner(
			strings.NewReader(
				string(data),
			),
		)

	for scanner.Scan() {
		fields :=
			strings.Fields(
				scanner.Text(),
			)

		if len(fields) < 2 {
			continue
		}

		key :=
			strings.TrimSuffix(
				fields[0],
				":",
			)

		if key != "MemTotal" &&
			key != "MemFree" &&
			key != "MemAvailable" {

			continue
		}

		value, err :=
			strconv.ParseUint(
				fields[1],
				10,
				64,
			)

		if err != nil {
			return NodeMemorySnapshot{},
				fmt.Errorf(
					"invalid %s value %q: %w",
					key,
					fields[1],
					err,
				)
		}

		multiplier :=
			uint64(1)

		if len(fields) >= 3 {
			switch strings.ToLower(
				fields[2],
			) {
			case "kb":
				multiplier = 1024

			case "b":
				multiplier = 1

			default:
				return NodeMemorySnapshot{},
					fmt.Errorf(
						"unsupported %s unit %q",
						key,
						fields[2],
					)
			}
		}

		if value >
			math.MaxUint64/
				multiplier {

			return NodeMemorySnapshot{},
				fmt.Errorf(
					"%s overflows bytes",
					key,
				)
		}

		values[key] =
			value *
				multiplier
	}

	if err := scanner.Err(); err != nil {
		return NodeMemorySnapshot{},
			fmt.Errorf(
				"scan /proc/meminfo: %w",
				err,
			)
	}

	requiredFields :=
		[]string{
			"MemTotal",
			"MemFree",
			"MemAvailable",
		}

	for _, required := range requiredFields {

		if _, ok :=
			values[required]; !ok {

			return NodeMemorySnapshot{},
				fmt.Errorf(
					"%s not found",
					required,
				)
		}
	}

	return NodeMemorySnapshot{
		TotalBytes: values["MemTotal"],

		FreeBytes: values["MemFree"],

		AvailableBytes: values["MemAvailable"],
	}, nil
}

func parseProcVMStat(
	data []byte,
) (NodeVMStatSnapshot, error) {
	values :=
		make(
			map[string]uint64,
		)

	scanner :=
		bufio.NewScanner(
			strings.NewReader(
				string(data),
			),
		)

	for scanner.Scan() {
		fields :=
			strings.Fields(
				scanner.Text(),
			)

		if len(fields) != 2 {
			continue
		}

		if fields[0] != "pgfault" &&
			fields[0] != "pgmajfault" {

			continue
		}

		value, err :=
			strconv.ParseUint(
				fields[1],
				10,
				64,
			)

		if err != nil {
			return NodeVMStatSnapshot{},
				fmt.Errorf(
					"invalid %s value %q: %w",
					fields[0],
					fields[1],
					err,
				)
		}

		values[fields[0]] =
			value
	}

	if err := scanner.Err(); err != nil {
		return NodeVMStatSnapshot{},
			fmt.Errorf(
				"scan /proc/vmstat: %w",
				err,
			)
	}

	requiredFields :=
		[]string{
			"pgfault",
			"pgmajfault",
		}

	for _, required := range requiredFields {

		if _, ok :=
			values[required]; !ok {

			return NodeVMStatSnapshot{},
				fmt.Errorf(
					"%s not found",
					required,
				)
		}
	}

	return NodeVMStatSnapshot{
		PageFaults: values["pgfault"],

		MajorPageFaults: values["pgmajfault"],
	}, nil
}

func populateNodeCPUProfile(
	profile *NodeResourceProfile,
	before NodeCPUStatSnapshot,
	after NodeCPUStatSnapshot,
) error {
	type counterPair struct {
		name   string
		before uint64
		after  uint64
		target *uint64
	}

	var rawUserDelta uint64
	var rawNiceDelta uint64

	pairs :=
		[]counterPair{
			{
				name:   "cpu_user",
				before: before.UserTicks,
				after:  after.UserTicks,
				target: &rawUserDelta,
			},
			{
				name:   "cpu_nice",
				before: before.NiceTicks,
				after:  after.NiceTicks,
				target: &rawNiceDelta,
			},
			{
				name:   "cpu_kernel",
				before: before.KernelTicks,
				after:  after.KernelTicks,
				target: &profile.CPUKernelDeltaTicks,
			},
			{
				name:   "cpu_idle",
				before: before.IdleTicks,
				after:  after.IdleTicks,
				target: &profile.CPUIdleDeltaTicks,
			},
			{
				name:   "cpu_iowait",
				before: before.IOWaitTicks,
				after:  after.IOWaitTicks,
				target: &profile.CPUIOWaitDeltaTicks,
			},
			{
				name:   "cpu_irq",
				before: before.IRQTicks,
				after:  after.IRQTicks,
				target: &profile.CPUIRQDeltaTicks,
			},
			{
				name:   "cpu_softirq",
				before: before.SoftIRQTicks,
				after:  after.SoftIRQTicks,
				target: &profile.CPUSoftIRQDeltaTicks,
			},
			{
				name:   "cpu_steal",
				before: before.StealTicks,
				after:  after.StealTicks,
				target: &profile.CPUStealDeltaTicks,
			},
			{
				name:   "cpu_guest",
				before: before.GuestTicks,
				after:  after.GuestTicks,
				target: &profile.CPUGuestDeltaTicks,
			},
			{
				name:   "cpu_guest_nice",
				before: before.GuestNiceTicks,
				after:  after.GuestNiceTicks,
				target: &profile.CPUGuestNiceDeltaTicks,
			},
		}

	for _, pair := range pairs {

		delta, ok :=
			counterDelta(
				pair.after,
				pair.before,
			)

		if !ok {
			return fmt.Errorf(
				"node_%s_counter_regressed",
				pair.name,
			)
		}

		*pair.target =
			delta
	}

	// Linux includes guest time in user time and guest_nice time in nice
	// time. Subtracting them makes the exported CPU modes mutually exclusive.
	if profile.CPUGuestDeltaTicks >
		rawUserDelta {

		return errors.New(
			"node_cpu_guest_exceeds_user",
		)
	}

	if profile.CPUGuestNiceDeltaTicks >
		rawNiceDelta {

		return errors.New(
			"node_cpu_guest_nice_exceeds_nice",
		)
	}

	profile.CPUUserDeltaTicks =
		rawUserDelta -
			profile.CPUGuestDeltaTicks

	profile.CPUNiceDeltaTicks =
		rawNiceDelta -
			profile.CPUGuestNiceDeltaTicks

	profile.CPUTotalDeltaTicks =
		sumNodeUint64(
			profile.CPUUserDeltaTicks,
			profile.CPUNiceDeltaTicks,
			profile.CPUKernelDeltaTicks,
			profile.CPUIdleDeltaTicks,
			profile.CPUIOWaitDeltaTicks,
			profile.CPUIRQDeltaTicks,
			profile.CPUSoftIRQDeltaTicks,
			profile.CPUStealDeltaTicks,
			profile.CPUGuestDeltaTicks,
			profile.CPUGuestNiceDeltaTicks,
		)

	if profile.CPUTotalDeltaTicks == 0 {
		return errors.New(
			"node_cpu_total_delta_zero",
		)
	}

	profile.AvailableCPUs =
		after.AvailableCPUs

	if profile.AvailableCPUs <= 0 {
		profile.AvailableCPUs =
			before.AvailableCPUs
	}

	if profile.AvailableCPUs <= 0 {
		return errors.New(
			"node_available_cpus_invalid",
		)
	}

	return nil
}

func populateNormalizedNodeCPUTimes(
	profile *NodeResourceProfile,
) {
	availableCPUTimeMs :=
		profile.ExecutionWallTimeMs *
			float64(
				profile.AvailableCPUs,
			)

	totalTicks :=
		float64(
			profile.CPUTotalDeltaTicks,
		)

	normalize :=
		func(
			ticks uint64,
		) (float64, float64) {
			share :=
				float64(ticks) /
					totalTicks

			return availableCPUTimeMs *
					share,
				share *
					100.0
		}

	profile.CPUUserDeltaMs,
		profile.CPUUserPercent =
		normalize(
			profile.CPUUserDeltaTicks,
		)

	profile.CPUNiceDeltaMs,
		profile.CPUNicePercent =
		normalize(
			profile.CPUNiceDeltaTicks,
		)

	profile.CPUKernelDeltaMs,
		profile.CPUKernelPercent =
		normalize(
			profile.CPUKernelDeltaTicks,
		)

	profile.CPUIdleDeltaMs,
		profile.CPUIdlePercent =
		normalize(
			profile.CPUIdleDeltaTicks,
		)

	profile.CPUIOWaitDeltaMs,
		profile.CPUIOWaitPercent =
		normalize(
			profile.CPUIOWaitDeltaTicks,
		)

	profile.CPUIRQDeltaMs,
		profile.CPUIRQPercent =
		normalize(
			profile.CPUIRQDeltaTicks,
		)

	profile.CPUSoftIRQDeltaMs,
		profile.CPUSoftIRQPercent =
		normalize(
			profile.CPUSoftIRQDeltaTicks,
		)

	profile.CPUStealDeltaMs,
		profile.CPUStealPercent =
		normalize(
			profile.CPUStealDeltaTicks,
		)

	profile.CPUGuestDeltaMs,
		profile.CPUGuestPercent =
		normalize(
			profile.CPUGuestDeltaTicks,
		)

	profile.CPUGuestNiceDeltaMs,
		profile.CPUGuestNicePercent =
		normalize(
			profile.CPUGuestNiceDeltaTicks,
		)
}

func isPerCPUName(
	value string,
) bool {
	if len(value) <= 3 ||
		!strings.HasPrefix(
			value,
			"cpu",
		) {

		return false
	}

	for _, character := range value[3:] {

		if character < '0' ||
			character > '9' {

			return false
		}
	}

	return true
}

func prefixNodeErrors(
	prefix string,
	values []string,
) []string {
	result :=
		make(
			[]string,
			0,
			len(values),
		)

	for _, value := range values {

		result =
			append(
				result,
				prefix+
					": "+
					value,
			)
	}

	return result
}

func averageNodeUint64(
	first uint64,
	second uint64,
) uint64 {
	return first/2 +
		second/2 +
		(first%2+second%2)/2
}

func sumNodeUint64(
	values ...uint64,
) uint64 {
	var result uint64

	for _, value := range values {

		result += value
	}

	return result
}
