package mab

import (
	"log"
	"strings"
	"sync"

	"github.com/serverledge-faas/serverledge/internal/config"
)

// BanditManager contains all the existing bandits (one for each known function)
type BanditManager struct {
	bandits   map[string]Policy // Nota: ora è map[string]Policy, non *UCB1Bandit
	mu        sync.RWMutex
	knownArms []string // This contains all founded architectures
}

var GlobalBanditManager *BanditManager

// InitBanditManager sets up the bandit manager
func InitBanditManager() {
	GlobalBanditManager = &BanditManager{
		bandits: make(map[string]Policy),
	}
}

// AddArmToAll once the system discover a new architecture, he has to add it to the available options
func (bm *BanditManager) AddArmToAll(arm string) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	bm.knownArms = append(bm.knownArms, arm)
	for _, bandit := range bm.bandits {
		bandit.InitArm(arm)
	}
}

// GetBandit returns (or creates) the bandit for a given function for all known architectures.
func (bm *BanditManager) GetBandit(functionName string) Policy {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	if bandit, exists := bm.bandits[functionName]; exists {
		return bandit
	}

	newBandit := bm.newBanditLocked(functionName)
	bm.bandits[functionName] = newBandit

	return newBandit
}

// newBanditLocked builds a policy initialized with every architecture currently
// known by the manager. The caller must hold bm.mu for writing.
//
// Keeping construction in a single helper is important for runtime transfer:
// the target policy can be fully initialized with a donor prior before it is
// published in bm.bandits and becomes visible to request handling.
func (bm *BanditManager) newBanditLocked(functionName string) Policy {
	configuredPolicy :=
		config.GetString(
			config.MAB_POLICY,
			string(UCB1),
		)

	log.Printf(
		"BanditManager newBandit: policy type: %s\n",
		configuredPolicy,
	)

	normalizedPolicy :=
		strings.ToLower(
			strings.TrimSpace(
				configuredPolicy,
			),
		)

	var newBandit Policy

	switch normalizedPolicy {
	case "linucb":
		alpha :=
			config.GetFloat(
				config.MAB_LINUCB_ALPHA,
				0.1,
			)

		newBandit =
			NewLinUCBDisjointPolicy(
				functionName,
				alpha,
			)

	default:
		newBandit =
			NewUCB1Bandit(
				functionName,
				config.GetFloat(
					config.MAB_UCB1_C,
					0.8,
				),
			)
	}

	for _, arm := range bm.knownArms {
		newBandit.InitArm(arm)
	}

	return newBandit
}
