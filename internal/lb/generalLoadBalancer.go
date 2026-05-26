package lb

import (
	"fmt"
	"log"
	"math/rand"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/lithammer/shortuuid"
	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/serverledge-faas/serverledge/internal/container"
	"github.com/serverledge-faas/serverledge/internal/function"
	"github.com/serverledge-faas/serverledge/internal/mab"
)

type GeneralLoadBalancer struct {
	mu sync.Mutex

	// mode is the architecture selection strategy: ‘RoundRobin’, “MAB” or ‘Random’.
	mode string

	// rings maintains a separate hash ring for each architecture tag.
	// The key is the node’s machine_tag (e.g. ‘x86’, “arm”, ‘x86_plus’).
	// The rings are created dynamically in AddTarget when
	// a previously unseen tag appears (lazy initialisation).
	rings map[string]*HashRing

	// archRRIndex is the index for Round Robin across different architectures.
	// It cycles through b.architectures to distribute requests
	// evenly across all available tags when the mode is RoundRobin.
	archRRIndex int

	// replicas is the number of virtual nodes per physical node in the hash ring.
	// More replicas = more even distribution, but more memory used.
	// Configurable via lb.replicas (default 128).
	replicas int

	// rrIndexes stores the current Round Robin index for each architecture.
	// It is a generalisation of the armRRIndex and x86RRIndex from the previous version:
	// instead of two fixed fields, it is a map that works with N architectures.
	// Used for Round Robin scheduling of nodes within a single ring (non-invoke requests).
	rrIndexes map[string]int // generalized idea of RR non-invoke request -> TODO: i have to check this with the prof

	// tagToArch is used to save the machine tag using a ring that is compatible with
	// the architecture
	tagToArch map[string]string

	// architectures is the sorted list of tags recognised by the balancer.
	// It is used for round-robin load balancing across architectures: Go does not guarantee a consistent
	// order when iterating over maps, so without this slice, round-robin
	// load balancing would be non-deterministic and experiments would not be reproducible.
	// It is updated in `AddTarget` whenever a new tag appears.
	architectures []string // slice of architectures' ring
}

// NewGeneralLoadBalancer Constructor
func NewGeneralLoadBalancer(targets []*middleware.ProxyTarget) *GeneralLoadBalancer {

	// REPLICAS is the number of times each physical node will appear in the hash ring. This is done to improve how
	// virtual nodes (i.e.: replicas of each physical node) are distributed over the ring, to reduce variation.
	REPLICAS := config.GetInt(config.REPLICAS, 128)
	log.Printf("Running ArchitectureAwareLB with %d replicas per node in the hash rings\n", REPLICAS)

	b := &GeneralLoadBalancer{
		rings:       make(map[string]*HashRing),
		archRRIndex: 0,
		replicas:    REPLICAS,
		rrIndexes:   make(map[string]int),
		tagToArch:   make(map[string]string),
	}

	b.mode = config.GetString(config.LB_MODE, RR)
	log.Printf("LB mode set to %s\n", b.mode)

	// to stay consistent with the old RoundRobinLoadBalancer, we'll still a single target list, that will contain all nodes,
	// both ARM and x86. We will now sort them into the respective hashRings.

	// Added the ability to add nodes regardless of the architecture.
	// TODO: I need to understand the dependency with the LoadBalancer RR, I think that this is enough
	for _, t := range targets {
		tag, ok := t.Meta["machine_tag"].(string)
		if !ok || tag == "" {
			log.Printf("[LB] Node %s has no machine_tag, skipping\n", t.Name)
			continue
		}
		b.AddTarget(t)
	}

	return b
}

// Next Used by Echo Proxy middleware to select the next target dynamically

/*
				Next() asks the MAB: ‘Which tag should I choose?’
	                → The MAB only knows the initialised arms
	                → It chooses from among them
	                → It returns a tag, e.g. ‘x86’
	            Next() looks for b.rings[‘x86’].Get(fun)
	                → It finds a node and returns it
*/
// TODO: modify the callee proxy and there modify the logic for 492 code response: "Architecture not supported"
func (b *GeneralLoadBalancer) Next(c echo.Context) *middleware.ProxyTarget {
	b.mu.Lock()
	defer b.mu.Unlock()

	if !isInvoke(c) {
		log.Printf("c NOT INVOKE: %s\n", c.Path())
		// fallback to round-robin

		// TODO: ask to prof
		// For now, I'll leave the handling of requests that aren't invocations—as
		// was the case with the previous LoadBalancer—to the generalized management of rings
		var candidate *middleware.ProxyTarget
		arch := b.selectArchitectureRR() // return a string value of the selected architectures
		if arch == "" {
			log.Printf("[LB] No available architecture for non-invoke request\n")
			return nil
		}

		// If there are no nodes available for the selected ring, return nil
		ring := b.rings[arch]
		if ring == nil {
			log.Printf("[LB] No available architecture for non-invoke request\n")
			return nil
		}

		idx := b.rrIndexes[arch]
		candidate = b.rings[arch].targetList[idx]
		b.rrIndexes[arch] = (idx + 1) % b.rings[arch].Size()

		return candidate
	}

	funcName := extractFunctionName(c)        // get function's name from request's URL
	fun, ok := function.GetFunction(funcName) // we use this to leverage cache before asking etcd
	if !ok {
		log.Printf("Dropping request for unknown fun '%s'\n", funcName)
		return nil
	}

	reqID := shortuuid.New()
	c.Request().Header.Set("Serverledge-MAB-Request-ID", reqID)

	var ctx *mab.Context = nil
	if b.mode == MAB {
		ctx = b.calculateSystemContext()           // memory snapshot for the MAB LinUCB
		mab.GlobalContextStorage.Store(reqID, ctx) // Cache it for LinUCB update
	}

	// There is a problem, a function can't be executed everywhere now
	// If only one architecture is supported skip the MAB and just use that
	targetArch := ""
	if len(fun.SupportedArchs) == 1 {
		targetArch = fun.SupportedArchs[0]
	} else if b.mode == MAB {
		// if both are supported, then use the MAB to select it
		bandit := mab.GlobalBanditManager.GetBandit(funcName)
		targetArch = bandit.SelectArm(ctx)
	} else if b.mode == RR { // RoundRobin
		// here the load balancer decides what architecture to use for this function
		targetArch = b.selectArchitectureRRForFunction(fun)
	} else {
		// Random load balancer for testing purposes
		targetArch = b.selectArchitectureRandom()
	}

	// once we selected an architecture, we'll use consistent hashing to select what node to use
	// The Get function will cycle through the hashRing to find a suitable node. If none is find we try to check if in
	// the other ring there is a suitable node for the function, to maximize chances of execution.
	var candidate *middleware.ProxyTarget

	// Prioritize the selected architectures
	// Cases in which the candidate selected based on the established framework is not available
	// 1. x86
	// 2. ARM
	// TODO: ask to prof if this is ok
	// Prioritize the selected tag
	if b.rings[targetArch] != nil {
		candidate = b.rings[targetArch].Get(fun)
	}

	// Fallback: iterates through the other rings in a deterministic order
	if candidate == nil {
		pattern := config.GetString(config.FUNCTION_TAG_PATTERN, "")
		for _, tag := range b.architectures {
			if tag == targetArch {
				continue // already tried
			}
			if matchesPattern(tag, pattern) && b.rings[tag].Size() > 0 {
				candidate = b.rings[tag].Get(fun)
				if candidate != nil {
					log.Printf("[LB] Fallback: using ring '%s'\n", tag)
					break
				}
			}
		}
	}

	if candidate != nil {
		freeMemoryMB := NodeMetrics.GetFreeMemory(candidate.Name) - fun.MemoryMB
		// Remove the memory that this function will use (this will then be updated again once the function is executed)
		freeCpu := NodeMetrics.metrics[candidate.Name].FreeCPU - fun.CPUDemand
		NodeMetrics.Update(candidate.Name, freeMemoryMB, 0, time.Now().Unix(), freeCpu)

	}
	return candidate
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

// The same of the original but with the update of load balancer
// TODO: Ask to prof if I can delete this function
// Deprecated
// This should only 	be used for tests or as a baseline in experiments.
// selectArchitecture checks the function's runtime to see what architecture it can support. Then it checks if any
// available node of the corresponding architecture is available. If the runtime supports both architecture, then we
// have a tie-break and select a node from the chosen list (arm or x86).
func (b *GeneralLoadBalancer) selectArchitecture(fun *function.Function) (string, error) {
	supportsArm := fun.SupportsArch(container.ARM)
	supportsX86 := fun.SupportsArch(container.X86)

	if supportsArm && supportsX86 {
		cacheValidity := 30 * time.Second // may be fine-tuned
		cacheEntry, ok := ArchitectureCacheLB.cache[fun.Name]

		// If we have a valid cache entry, we try to use it
		expiry := time.Unix(cacheEntry.Timestamp, 0).Add(cacheValidity)
		if ok && time.Now().Before(expiry) {
			// Check if the cached architecture has available nodes
			if (cacheEntry.Arch == container.ARM && b.rings[container.ARM].Size() > 0) ||
				(cacheEntry.Arch == container.X86 && b.rings[container.X86].Size() > 0) {
				// If the cached architecture is still valid and has available nodes, use it
				cacheEntry.Timestamp = time.Now().Unix() // Update timestamp
				ArchitectureCacheLB.cache[fun.Name] = cacheEntry
				return cacheEntry.Arch, nil
			}

		}

		// Tie-breaking: if both architectures are supported, prefer ARM if available (less energy consumption), otherwise x86.
		// This will also be the fallback if the cached decision is not usable.
		var chosenArch string
		if b.rings[container.ARM].Size() > 0 {
			chosenArch = container.ARM
		} else if b.rings[container.X86].Size() > 0 {
			chosenArch = container.X86
		} else {
			return "", fmt.Errorf("no available nodes for either ARM or x86")
		}

		// Update cache
		newCacheEntry := ArchitectureCacheEntry{
			Arch:      chosenArch,
			Timestamp: time.Now().Unix(),
		}
		ArchitectureCacheLB.cache[fun.Name] = newCacheEntry

		return chosenArch, nil
	}

	if supportsArm {
		if b.rings[container.ARM].Size() > 0 {
			return container.ARM, nil
		}
		return "", fmt.Errorf("no ARM nodes available")
	}

	if supportsX86 {
		if b.rings[container.X86].Size() > 0 {
			return container.X86, nil
		}
		return "", fmt.Errorf("no x86 nodes available")
	}

	return "", fmt.Errorf("function does not support any available architecture")
}

// selectArchitectureRR selects the architecture using a Round Robin policy.
func (b *GeneralLoadBalancer) selectArchitectureRR() string {
	if len(b.architectures) == 0 {
		return ""
	}
	// TODO: check if the comment is still true, but I don't think it is
	// This is just a function to use as a baseline for the LB. It should actually implement checks over the rings dimension.
	// i.e.: it cannot select ARM/X86 "blindly", it should check if we have at least one node for that architecture.
	var selected string
	selected = ""

	// I take the previous index value, which was incremented based on the architecture, and use it
	// to determine the architecture to be considered for the Round Robin selection
	selected = b.architectures[b.archRRIndex]
	if b.rings[selected].Size() > 0 {
		// If the corresponding node within the structure is valid, then I'll consider it for selection
		// and update the index value for the next iteration
		b.archRRIndex = (b.archRRIndex + 1) % len(b.architectures)
	} else {
		// If the selection is invalid, then I need to check which ring is available in terms of the presence of nodes;
		// once I have that information, I save the index value to use for the next iteration
		selected = ""
		for i := 0; i < len(b.architectures); i++ {
			index := (b.archRRIndex + i) % len(b.architectures)
			selected = b.architectures[index]
			if b.rings[selected].Size() > 0 {
				b.archRRIndex = (index + 1) % len(b.architectures)
				break
			}
		}

	}
	return selected
}

// AddTarget Echo requires this method for dynamic load-balancing. It simply inserts a new node in the respective ring.
func (b *GeneralLoadBalancer) AddTarget(t *middleware.ProxyTarget) bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	nodeInfo := GetSingleTargetInfo(t)
	// Every time we add a node, we set the information about its available memory
	if nodeInfo != nil {
		// Update will update the freeMemory only if the information in nodeInfo is fresher than what we
		// already have in the NodeMetrics cache.
		NodeMetrics.Update(
			t.Name,
			nodeInfo.AvailableMemory,
			nodeInfo.TotalMemory,
			nodeInfo.LastUpdateTime,
			nodeInfo.TotalCPU-nodeInfo.UsedCPU)
	}

	// Depending on the architecture, the node is placed in the correct ring.
	// If the ring does not yet exist, it is created (Celebrinbor join the chat)
	tag := t.Meta["machine_tag"].(string)
	if _, exists := b.rings[tag]; !exists {
		b.rings[tag] = NewHashRing(b.replicas)
		b.architectures = append(b.architectures, tag)
		b.rrIndexes[tag] = 0
		// Update bandit arm
		mab.GlobalBanditManager.AddArmToAll(tag)
	}

	fmt.Printf("[LB] Adding target %s\n", tag)
	arch := t.Meta["arch"].(string)
	b.tagToArch[tag] = arch
	b.rings[tag].Add(t)

	return true
}

// RemoveTarget Echo requires this method to remove a target by name
func (b *GeneralLoadBalancer) RemoveTarget(name string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	delete(NodeMetrics.metrics, name) // this is no longer needed

	for _, tag := range b.architectures {
		if b.rings[tag].RemoveByName(name) {
			return true
		}
	}

	return false
}

// selectArchitectureRRForFunction is used to determine the next node to be considered when selecting how to execute
// the function, based on the specified and indicated architecture
func (b *GeneralLoadBalancer) selectArchitectureRRForFunction(fun *function.Function) string {
	if len(b.architectures) == 0 {
		return ""
	}

	for i := 0; i < len(b.architectures); i++ {
		index := (b.archRRIndex + i) % len(b.architectures)
		tag := b.architectures[index]
		arch, ok := b.tagToArch[tag]
		if !ok {
			continue
		}
		if fun.SupportsArch(arch) && b.rings[tag].Size() > 0 {
			b.archRRIndex = (index + 1) % len(b.architectures)
			return tag
		}
	}
	return ""
}

// matchesPattern is used to find the next node to call based on the specified function criteria,
// whether a node is compatible with the specified architecture
func matchesPattern(tag string, pattern string) bool {
	if pattern == "" {
		return true // no pattern = any tag is acceptable
	}
	matched, err := regexp.MatchString(pattern, tag)
	if err != nil {
		return false
	}
	return matched
}

func (b *GeneralLoadBalancer) selectArchitectureRandom() string {
	// Seed the random number generator if needed, though global rand is usually fine for simple LB
	// rand.Seed(time.Now().UnixNano())
	if len(b.architectures) == 0 {
		return ""
	}
	index := rand.Intn(len(b.architectures))
	return b.architectures[index]
}
