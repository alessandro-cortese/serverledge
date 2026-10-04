package main

// Adaptation of the vSwarm Auth standalone benchmark.
// Original benchmark: vSwarm/benchmarks/auth/go/server.go
// Copyright (c) 2022 EASE lab — MIT License.
//
// Adaptation policy:
// - original token-based authorization logic preserved;
// - original policy-generation logic preserved;
// - gRPC, AWS Lambda glue and distributed tracing removed;
// - Serverledge handler added.

import (
	"fmt"
	"runtime"
	"strings"

	"github.com/serverledge-faas/serverledge/serverledge"
)

type Statement struct {
	Action   string `json:"action"`
	Effect   string `json:"effect"`
	Resource string `json:"resource"`
}

type PolicyDocument struct {
	Version   string      `json:"version"`
	Statement []Statement `json:"statement"`
}

type AuthContext struct {
	StringKey  string `json:"string_key"`
	NumberKey  int    `json:"number_key"`
	BooleanKey bool   `json:"boolean_key"`
}

type AuthResponse struct {
	PrincipalID    string         `json:"principal_id"`
	PolicyDocument PolicyDocument `json:"policy_document"`
	Context        AuthContext    `json:"context"`
}

func generatePolicy(principalID, effect, resource string) AuthResponse {
	var response AuthResponse

	response.PrincipalID = principalID

	if effect != "" && resource != "" {
		var policy PolicyDocument
		policy.Version = "2012-10-17"

		var statement Statement
		statement.Action = "execute-api:Invoke"
		statement.Effect = effect
		statement.Resource = resource

		policy.Statement = []Statement{statement}
		response.PolicyDocument = policy
	}

	response.Context = AuthContext{
		StringKey:  "stringval",
		NumberKey:  123,
		BooleanKey: true,
	}

	return response
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	token := "allow"
	if value, ok := params["token"].(string); ok {
		value = strings.TrimSpace(value)
		if value != "" {
			token = value
		}
	}

	const fakeMethodARN = "arn:aws:execute-api:{regionId}:{accountId}:{apiId}/{stage}/{httpVerb}/[{resource}/[{child-resources}]]"

	var message string
	var response AuthResponse

	switch token {
	case "allow":
		response = generatePolicy("user", "Allow", fakeMethodARN)
		message = fmt.Sprintf("%+v", response)

	case "deny":
		response = generatePolicy("user", "Deny", fakeMethodARN)
		message = fmt.Sprintf("%+v", response)

	case "unauthorized":
		message = "Unauthorized"

	default:
		message = "Error: Invalid token"
	}

	return map[string]interface{}{
		"benchmark": "vswarm-auth",
		"token":     token,
		"response":  message,
		"arch":      runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
