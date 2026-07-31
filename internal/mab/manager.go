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

// GetBandit returns (or creates) the bandit for a given function for all known architectures
func (bm *BanditManager) GetBandit(functionName string) Policy {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	if _, exists := bm.bandits[functionName]; !exists {
		// Read policy from config
		policyType := config.GetString(config.MAB_POLICY, "UCB1")
		log.Printf("BanditManager GetBandit: policy type: %s\n", policyType)

		configuredPolicy :=
			config.GetString(
				config.MAB_POLICY,
				string(UCB1),
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

		// Init arm of all known architectures
		for _, arm := range bm.knownArms {
			newBandit.InitArm(arm)
		}

		bm.bandits[functionName] = newBandit
	}
	return bm.bandits[functionName]
}
