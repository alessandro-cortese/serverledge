package lb

import (
	"log"
	"math/rand"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/lithammer/shortuuid"
	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/serverledge-faas/serverledge/internal/function"
	"github.com/serverledge-faas/serverledge/internal/mab"
)

type GeneralLoadBalancer struct {
	mu sync.Mutex

	// mode is the architecture/tag selection strategy: "RoundRobin", "MAB" or "Random".
	mode string

	// rings maintains a separate hash ring for each architecture tag.
	// The key is the node’s machine_tag (e.g. ‘x86’, “arm”, ‘x86_plus’).
	// The rings are created dynamically in AddTarget when
	// a previously unseen tag appears (lazy initialisation).
	rings map[string]*HashRing

	// archRRIndex is the index for Round Robin across different architectures/tags.
	// It cycles through b.architectures to distribute requests
	// evenly across all available tags when the mode is RoundRobin.
	//
	// NOTE: the field name is kept for continuity with the old implementation,
	// but the values in b.architectures are now machine tags, not necessarily
	// physical CPU architectures.
	archRRIndex int

	// replicas is the number of virtual nodes per physical node in the hash ring.
	// More replicas = more even distribution, but more memory used.
	// Configurable via lb.replicas (default 128).
	replicas int

	// rrIndexes stores the current Round Robin index for each architecture/tag.
	// It is a generalisation of the armRRIndex and x86RRIndex from the previous version:
	// instead of two fixed fields, it is a map that works with N architectures/tags.
	//
	// Used for Round Robin scheduling of nodes within a single ring, especially
	// for non-invoke requests where there is no function-specific hardware requirement.
	rrIndexes map[string]int

	// tagToArch is used to save which physical/runtime architecture corresponds
	// to a machine tag.
	// This allows the balancer to build rings using machine_tag while still checking
	// whether the function supports the physical architecture required for execution.
	tagToArch map[string]string

	// architectures is the sorted/deterministic list of tags recognised by the balancer.
	// It is used for round-robin load balancing across architectures/tags:
	// Go does not guarantee a consistent order when iterating over maps, so without
	// this slice, round-robin load balancing would be non-deterministic and experiments
	// would not be reproducible.
	//
	// It is updated in AddTarget whenever a new tag appears.
	architectures []string
}

// NewGeneralLoadBalancer Constructor
func NewGeneralLoadBalancer(targets []*middleware.ProxyTarget) *GeneralLoadBalancer {

	// REPLICAS is the number of times each physical node will appear in the hash ring.
	// This is done to improve how virtual nodes (i.e.: replicas of each physical node)
	// are distributed over the ring, to reduce variation.
	replicas := config.GetInt(config.REPLICAS, 128)
	log.Printf("Running ArchitectureAwareLB with %d replicas per node in the hash rings\n", replicas)

	b := &GeneralLoadBalancer{
		rings:       make(map[string]*HashRing),
		archRRIndex: 0,
		replicas:    replicas,
		rrIndexes:   make(map[string]int),
		tagToArch:   make(map[string]string),
	}

	b.mode = config.GetString(config.LB_MODE, RR)
	log.Printf("LB mode set to %s\n", b.mode)

	// To stay consistent with the old RoundRobinLoadBalancer, we still receive a
	// single target list that contains all nodes.
	//
	// The difference with the old implementation is internal:
	// instead of splitting the nodes only into fixed ARM/x86 rings, we now sort them
	// into dynamically created hash rings, one for each machine_tag.
	//
	// Added the ability to add nodes regardless of the architecture/tag.
	for _, t := range targets {
		tag := getTargetMachineTag(t)
		if tag == "" {
			log.Printf("[LB] Node %s has no machine_tag/arch, skipping\n", t.Name)
			continue
		}
		b.AddTarget(t)
	}

	return b
}

// Next Used by Echo Proxy middleware to select the next target dynamically.

/*
	Next() asks the MAB: "Which tag should I choose?"
	    → The MAB only knows the initialized arms
	    → It chooses from among them
	    → It returns a tag, e.g. "x86_plus"

	Next() looks for b.rings["x86_plus"].Get(fun)
	    → It finds a node and returns it

	Important:
	- Rings are indexed by machine_tag.
	- Function compatibility is still checked through the physical architecture
	  associated with each tag, stored in b.tagToArch.
*/

func (b *GeneralLoadBalancer) Next(c echo.Context) *middleware.ProxyTarget {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !isInvoke(c) {
		log.Printf("c NOT INVOKE: %s\n", c.Path())

		// Fallback to Round Robin.
		//
		// For now, we keep the previous behaviour for requests that are not invocations:
		// since these requests do not refer to a specific function, they do not have
		// function-level hardware constraints. Therefore, we select one available tag
		// using the generalized Round Robin logic over the rings.
		tag := b.selectArchitectureRR()
		if tag == "" {
			log.Printf("[LB] No available tag for non-invoke request\n")
			return nil
		}

		// If there are no nodes available for the selected ring, return nil.
		ring := b.rings[tag]
		if ring == nil || ring.Size() == 0 {
			log.Printf("[LB] Selected tag '%s' has no available targets for non-invoke request\n", tag)
			return nil
		}

		idx := b.rrIndexes[tag] % ring.Size()
		candidate := ring.targetList[idx]
		b.rrIndexes[tag] = (idx + 1) % ring.Size()

		return candidate
	}

	funcName := extractFunctionName(c)        // get function's name from request's URL
	fun, ok := function.GetFunction(funcName) // we use this to leverage cache before asking etcd
	if !ok {
		log.Printf("Dropping request for unknown function '%s'\n", funcName)
		return nil
	}

	reqID := shortuuid.New()
	c.Request().Header.Set("Serverledge-MAB-Request-ID", reqID)

	var ctx *mab.Context = nil
	if b.mode == MAB {
		ctx = b.calculateSystemContext()           // memory snapshot for the MAB/LinUCB
		mab.GlobalContextStorage.Store(reqID, ctx) // Cache it for LinUCB update
	}

	// A function cannot necessarily be executed everywhere.
	//
	// With machine_tag, the selected value must be a tag/ring key, not directly
	// a physical architecture such as amd64 or arm64. For this reason we first
	// compute the set of tags that are compatible with the function.
	compatibleTags := b.compatibleTagsForFunction(fun)
	if len(compatibleTags) == 0 {
		log.Printf("[LB] No compatible machine tag available for function '%s'\n", funcName)
		setHardwareNotSupported(c, funcName)
		return nil
	}

	targetTag := b.selectTargetTag(funcName, fun, compatibleTags, ctx)
	if targetTag == "" {
		log.Printf("[LB] Unable to select a compatible tag for function '%s'\n", funcName)
		setHardwareNotSupported(c, funcName)
		return nil
	}

	// Once we selected a tag, we use consistent hashing to select the concrete node.
	// The Get function will cycle through the hash ring to find a suitable node.
	// If none is found in the selected ring, we try the other compatible rings in a
	// deterministic order to maximize chances of execution.
	candidate := b.getCandidateFromTag(targetTag, fun)
	if candidate == nil {
		candidate = b.getCandidateFromFallbackTags(targetTag, compatibleTags, fun)
	}

	if candidate == nil {
		log.Printf("[LB] No candidate available for function '%s' after fallback\n", funcName)
		setHardwareNotSupported(c, funcName)
		return nil
	}

	// Remove the memory that this function will use.
	// This will then be updated again once the function is executed and the node
	// reports its real free memory through response headers.
	freeMemoryMB := NodeMetrics.GetFreeMemory(candidate.Name) - fun.MemoryMB
	freeCpu := 0.0
	if metric, ok := NodeMetrics.GetMetric(candidate.Name); ok {
		freeCpu = metric.FreeCPU - fun.CPUDemand
	}
	NodeMetrics.Update(candidate.Name, freeMemoryMB, 0, time.Now().Unix(), freeCpu)

	candidateTag := ""
	if candidate.Meta != nil {
		if tag, ok := candidate.Meta["machine_tag"].(string); ok {
			candidateTag = tag
		}
	}

	log.Printf(
		"[LB] Selected target for function '%s': target_tag=%s candidate=%s candidate_tag=%s\n",
		funcName,
		targetTag,
		candidate.Name,
		candidateTag,
	)

	return candidate
}

// selectTargetTag selects the machine tag to prioritize for an invoke request.
// The selection domain is already restricted to compatibleTags.
func (b *GeneralLoadBalancer) selectTargetTag(funcName string, fun *function.Function, compatibleTags []string, ctx *mab.Context) string {
	if len(compatibleTags) == 0 {
		return ""
	}

	// If only one tag is compatible, skip MAB/RR/Random and use that tag directly.
	if len(compatibleTags) == 1 {
		return compatibleTags[0]
	}

	switch b.mode {
	case MAB:
		log.Printf(
			"[LB][MAB] event=before_select function=%s compatible_tags=%v has_context=%t\n",
			funcName,
			compatibleTags,
			ctx != nil,
		)

		bandit := mab.GlobalBanditManager.GetBandit(funcName)
		selectedTag := bandit.SelectArm(ctx)

		log.Printf(
			"[LB][MAB] event=after_select function=%s selected_tag=%s compatible_tags=%v\n",
			funcName,
			selectedTag,
			compatibleTags,
		)

		if containsString(compatibleTags, selectedTag) {
			// Update archRRIndex to maintain consistency if the mode changes
			for i, tag := range b.architectures {
				if tag == selectedTag {
					b.archRRIndex = (i + 1) % len(b.architectures)
					break
				}
			}
			return selectedTag
		}

		log.Printf(
			"[LB][MAB] event=incompatible_selection function=%s selected_tag=%s compatible_tags=%v fallback=rr\n",
			funcName,
			selectedTag,
			compatibleTags,
		)

		return b.selectArchitectureRRFrom(compatibleTags)

	case RR:
		// RoundRobin baseline: select only among tags that are compatible with this function.
		return b.selectArchitectureRRFrom(compatibleTags)

	default:
		// Random load balancer for testing purposes.
		// It must still select only among compatible tags.
		return b.selectArchitectureRandomFrom(compatibleTags)
	}
}

// extractFunctionName retrieves the function's name by parsing the request's URL.
func extractFunctionName(c echo.Context) string {
	path := c.Request().URL.Path

	const prefix = "/invoke/"
	if !strings.HasPrefix(path, prefix) {
		return "" // not an invocation
	}

	return path[len(prefix):]
}

// compatibleTagsForFunction returns the machine tags that can currently execute fun.
//
// A tag is compatible if:
//  1. the corresponding ring exists and has at least one target;
//  2. the tag is mapped to a physical/runtime architecture;
//  3. the function supports that physical/runtime architecture;
//  4. the tag matches the configured tag pattern, if any.
//
// The tag pattern is read from the Function itself.
// If the function has no TagPattern, every machine_tag is accepted.
func (b *GeneralLoadBalancer) compatibleTagsForFunction(fun *function.Function) []string {
	compatibleTags := make([]string, 0)

	pattern := getFunctionTagPattern(fun)

	for _, tag := range b.architectures {
		ring := b.rings[tag]
		if ring == nil || ring.Size() == 0 {
			continue
		}

		arch, ok := b.tagToArch[tag]
		if !ok || arch == "" {
			log.Printf("[LB] Missing physical architecture for tag '%s'\n", tag)
			continue
		}

		if !fun.SupportsArch(arch) {
			continue
		}

		if !matchesPattern(tag, pattern) {
			continue
		}
		compatibleTags = append(compatibleTags, tag)

		log.Printf(
			"[LB] Compatible tag found: function=%s tag=%s arch=%s pattern=%q ring_size=%d\n",
			fun.Name, tag, arch, pattern, ring.Size(),
		)
	}

	log.Printf("[LB] Compatible tags for function '%s': %v\n", fun.Name, compatibleTags)
	return compatibleTags
}

// selectArchitectureRRFrom performs Round Robin only over the supplied compatible tags.
// It preserves the deterministic global order stored in b.architectures.
func (b *GeneralLoadBalancer) selectArchitectureRRFrom(tags []string) string {
	if len(tags) == 0 || len(b.architectures) == 0 {
		return ""
	}

	for i := 0; i < len(b.architectures); i++ {
		index := (b.archRRIndex + i) % len(b.architectures)
		tag := b.architectures[index]

		if !containsString(tags, tag) {
			continue
		}

		ring := b.rings[tag]
		if ring == nil || ring.Size() == 0 {
			continue
		}

		b.archRRIndex = (index + 1) % len(b.architectures)
		return tag
	}

	return ""
}

// selectArchitectureRRForFunction is used to determine the next tag to be considered
// when selecting where to execute the function.
//
// It keeps the old method name for compatibility with previous code/tests, but now
// returns a machine_tag, not necessarily a physical architecture.
func (b *GeneralLoadBalancer) selectArchitectureRRForFunction(fun *function.Function) string {
	return b.selectArchitectureRRFrom(b.compatibleTagsForFunction(fun))
}

// selectArchitectureRR selects the architecture/tag using a Round Robin policy.
//
// This is used for non-invoke requests, where there is no specific function and
// therefore no function-specific compatibility filter. It still checks that the
// selected ring has at least one node available.
func (b *GeneralLoadBalancer) selectArchitectureRR() string {
	if len(b.architectures) == 0 {
		return ""
	}

	for i := 0; i < len(b.architectures); i++ {
		index := (b.archRRIndex + i) % len(b.architectures)
		tag := b.architectures[index]

		ring := b.rings[tag]
		if ring == nil || ring.Size() == 0 {
			continue
		}

		b.archRRIndex = (index + 1) % len(b.architectures)
		return tag
	}

	return ""
}

// selectArchitectureRandomFrom randomly selects one tag from the supplied compatible tags.
func (b *GeneralLoadBalancer) selectArchitectureRandomFrom(tags []string) string {
	if len(tags) == 0 {
		return ""
	}

	index := rand.Intn(len(tags))
	return tags[index]
}

// selectArchitectureRandom keeps the old method name for non-invoke/tests.
// It randomly selects among all known tags, without function-specific filtering.
func (b *GeneralLoadBalancer) selectArchitectureRandom() string {
	if len(b.architectures) == 0 {
		return ""
	}

	index := rand.Intn(len(b.architectures))
	return b.architectures[index]
}

// AddTarget Echo requires this method for dynamic load-balancing.
// It simply inserts a new node in the respective ring.
func (b *GeneralLoadBalancer) AddTarget(t *middleware.ProxyTarget) bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Depending on the machine_tag, the node is placed in the correct ring.
	// If the ring does not yet exist, it is created lazily.
	tag := getTargetMachineTag(t)
	if tag == "" {
		log.Printf("[LB] Cannot add target %s: missing machine_tag and arch metadata\n", t.Name)
		return false
	}

	arch := getTargetArch(t)
	if arch == "" {
		log.Printf("[LB] Cannot add target %s: missing arch metadata\n", t.Name)
		return false
	}

	NodeMetrics.UpdateCostProfile(
		t.Name,
		getTargetCostFactor(t),
		getTargetEnergyFactor(t),
	)

	nodeInfo := GetSingleTargetInfo(t)
	// Every time we add a node, we set the information about its available memory.
	if nodeInfo != nil {
		// Update will update the freeMemory only if the information in nodeInfo is fresher
		// than what we already have in the NodeMetrics cache.
		NodeMetrics.Update(
			t.Name,
			nodeInfo.AvailableMemory,
			nodeInfo.TotalMemory,
			nodeInfo.LastUpdateTime,
			nodeInfo.TotalCPU-nodeInfo.UsedCPU,
		)

		NodeMetrics.UpdateCostProfile(
			t.Name,
			firstPositiveFloat64(nodeInfo.CostFactor, getTargetCostFactor(t)),
			firstPositiveFloat64(nodeInfo.EnergyFactor, getTargetEnergyFactor(t)),
		)
	}

	if _, exists := b.rings[tag]; !exists {
		b.rings[tag] = NewHashRing(b.replicas)
		b.architectures = append(b.architectures, tag)
		b.rrIndexes[tag] = 0

		// Update MAB arms dynamically when a new tag appears.
		log.Printf(
			"[LB][MAB] event=new_machine_tag_discovered tag=%s arch=%s action=create_ring_and_add_mab_arm\n",
			tag,
			arch,
		)

		mab.GlobalBanditManager.AddArmToAll(tag)
	}

	log.Printf("[LB] Adding target %s\n", tag)

	b.tagToArch[tag] = arch
	b.rings[tag].Add(t)

	return true
}

// RemoveTarget Echo requires this method to remove a target by name.
func (b *GeneralLoadBalancer) RemoveTarget(name string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	delete(NodeMetrics.metrics, name) // this is no longer needed

	for _, tag := range b.architectures {
		ring := b.rings[tag]
		if ring == nil {
			continue
		}

		if ring.RemoveByName(name) {
			return true
		}
	}

	return false
}

// getCandidateFromTag tries to select a concrete target from a specific ring/tag.
func (b *GeneralLoadBalancer) getCandidateFromTag(
	tag string,
	fun *function.Function,
) *middleware.ProxyTarget {
	if tag == "" {
		return nil
	}

	ring := b.rings[tag]
	if ring == nil || ring.Size() == 0 {
		return nil
	}

	return ring.Get(fun)
}

// getCandidateFromFallbackTags iterates through the other compatible rings in a
// deterministic order and returns the first candidate that can execute the function.
func (b *GeneralLoadBalancer) getCandidateFromFallbackTags(selectedTag string, compatibleTags []string, fun *function.Function) *middleware.ProxyTarget {
	for _, tag := range b.architectures {
		if tag == selectedTag {
			continue // already tried
		}

		if !containsString(compatibleTags, tag) {
			continue
		}

		candidate := b.getCandidateFromTag(tag, fun)
		if candidate != nil {
			log.Printf("[LB] Fallback: using ring '%s'\n", tag)
			return candidate
		}
	}

	return nil
}

// getTargetMachineTag returns the machine_tag associated with a target.
//
// Backward compatibility:
// if machine_tag is missing, the function falls back to arch.
// This preserves the old behaviour where one ring was created for ARM and one for x86.
func getTargetMachineTag(t *middleware.ProxyTarget) string {
	if t == nil || t.Meta == nil {
		return ""
	}

	if tag, ok := t.Meta["machine_tag"].(string); ok && tag != "" {
		return tag
	}

	if arch, ok := t.Meta["arch"].(string); ok && arch != "" {
		return arch
	}

	return ""
}

// getTargetArch returns the physical/runtime architecture associated with a target.
func getTargetArch(t *middleware.ProxyTarget) string {
	if t == nil || t.Meta == nil {
		return ""
	}

	if arch, ok := t.Meta["arch"].(string); ok && arch != "" {
		return arch
	}

	return ""
}

func firstPositiveFloat64(values ...float64) float64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 1.0
}

func getTargetCostFactor(t *middleware.ProxyTarget) float64 {
	if t == nil || t.Meta == nil {
		return 1.0
	}

	switch value := t.Meta["cost_factor"].(type) {
	case float64:
		return firstPositiveFloat64(value)
	case float32:
		return firstPositiveFloat64(float64(value))
	case int:
		return firstPositiveFloat64(float64(value))
	case int64:
		return firstPositiveFloat64(float64(value))
	case string:
		parsed, err := strconv.ParseFloat(value, 64)
		if err == nil {
			return firstPositiveFloat64(parsed)
		}
	}

	return 1.0
}

func getTargetEnergyFactor(t *middleware.ProxyTarget) float64 {
	if t == nil || t.Meta == nil {
		return 1.0
	}

	switch value := t.Meta["energy_factor"].(type) {
	case float64:
		return firstPositiveFloat64(value)
	case float32:
		return firstPositiveFloat64(float64(value))
	case int:
		return firstPositiveFloat64(float64(value))
	case int64:
		return firstPositiveFloat64(float64(value))
	case string:
		parsed, err := strconv.ParseFloat(value, 64)
		if err == nil {
			return firstPositiveFloat64(parsed)
		}
	}

	return 1.0
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

// matchesPattern is used to check whether a node tag is compatible with the specified
// function criteria.
//
// Empty pattern means that any tag is acceptable.
func matchesPattern(tag string, pattern string) bool {
	return mab.MachineTagMatchesRequirement(tag, pattern)
}

func getFunctionTagPattern(fun *function.Function) string {
	if fun == nil {
		return ""
	}

	return fun.TagPattern
}

func normalizeTagPattern(pattern string) string {
	if pattern == "" {
		return ""
	}

	// If it already looks like a regex, keep it.
	if strings.HasPrefix(pattern, "^") ||
		strings.HasSuffix(pattern, "$") ||
		strings.Contains(pattern, ".*") {
		return pattern
	}

	escaped := regexp.QuoteMeta(pattern)
	escaped = strings.ReplaceAll(escaped, `\*`, ".*")

	return "^" + escaped + "$"
}
