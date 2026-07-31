package mab

import (
	"testing"

	"github.com/serverledge-faas/serverledge/internal/config"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBanditManagerCreatesConfiguredPolicy(
	t *testing.T,
) {
	tests := []struct {
		name             string
		configuredPolicy string
		expectedType     BanditType
	}{
		{
			name:             "classic UCB1",
			configuredPolicy: "UCB1",
			expectedType:     UCB1,
		},
		{
			name:             "LinUCB",
			configuredPolicy: "LinUCB",
			expectedType:     LinUCB,
		},
		{
			name:             "unknown policy falls back to UCB1",
			configuredPolicy: "unknown-policy",
			expectedType:     UCB1,
		},
	}

	for _, test := range tests {
		t.Run(
			test.name,
			func(t *testing.T) {
				viper.Reset()
				t.Cleanup(viper.Reset)

				viper.Set(
					config.MAB_POLICY,
					test.configuredPolicy,
				)

				manager :=
					&BanditManager{
						bandits: make(
							map[string]Policy,
						),
						knownArms: []string{
							"arm-a",
							"arm-b",
						},
					}

				policy :=
					manager.GetBandit(
						"test-function",
					)

				require.NotNil(
					t,
					policy,
				)

				assert.Equal(
					t,
					test.expectedType,
					policy.GetType(),
				)

				switch typedPolicy :=
					policy.(type) {

				case *UCB1Bandit:
					assert.Contains(
						t,
						typedPolicy.Arms,
						"arm-a",
					)

					assert.Contains(
						t,
						typedPolicy.Arms,
						"arm-b",
					)

				case *LinUCBDisjointPolicy:
					assert.Contains(
						t,
						typedPolicy.Arms,
						"arm-a",
					)

					assert.Contains(
						t,
						typedPolicy.Arms,
						"arm-b",
					)

				default:
					t.Fatalf(
						"unexpected policy implementation: %T",
						policy,
					)
				}
			},
		)
	}
}
