package mab

import (
	"log"
	"regexp"
	"strings"
)

type MachineTagInfo struct {
	FullTag        string
	BaseTag        string
	Architecture   string
	Specialization string
	Capabilities   []string
}

// ParseMachineTag parses tags such as:
//
//	x86
//	x86-tiny
//	arm-large
//	x86-tiny/gpu
//	x86-large/gpu/nvidia
//
// Grammar:
//
//	<architecture>[-<specialization>][/<capability1>/<capability2>/...]
func ParseMachineTag(tag string) MachineTagInfo {
	normalized := strings.TrimSpace(strings.ToLower(tag))
	if normalized == "" {
		return MachineTagInfo{
			FullTag:        tag,
			BaseTag:        "unknown",
			Architecture:   "unknown",
			Specialization: "default",
			Capabilities:   []string{},
		}
	}

	parts := strings.Split(normalized, "/")
	baseTag := strings.TrimSpace(parts[0])
	if baseTag == "" {
		baseTag = "unknown"
	}

	capabilities := make([]string, 0, len(parts)-1)
	for _, rawCapability := range parts[1:] {
		capability := strings.TrimSpace(rawCapability)
		if capability != "" {
			capabilities = append(capabilities, capability)
		}
	}

	architecture := baseTag
	specialization := "default"

	baseParts := strings.SplitN(baseTag, "-", 2)
	if len(baseParts) == 2 {
		architecture = strings.TrimSpace(baseParts[0])
		specialization = strings.TrimSpace(baseParts[1])
	}

	if architecture == "" {
		architecture = "unknown"
	}
	if specialization == "" {
		specialization = "default"
	}

	return MachineTagInfo{
		FullTag:        normalized,
		BaseTag:        baseTag,
		Architecture:   architecture,
		Specialization: specialization,
		Capabilities:   capabilities,
	}
}

func HasCapability(machineTag string, requiredCapability string) bool {
	info := ParseMachineTag(machineTag)
	requiredCapability = strings.ToLower(strings.TrimSpace(requiredCapability))

	if requiredCapability == "" {
		return true
	}
	if info.FullTag == requiredCapability {
		return true
	}
	if info.BaseTag == requiredCapability {
		return true
	}
	if info.Architecture == requiredCapability {
		return true
	}
	if info.Specialization == requiredCapability {
		return true
	}

	for _, capability := range info.Capabilities {
		if capability == requiredCapability {
			return true
		}
	}

	return false
}

// MachineTagMatchesRequirement checks whether a node machine_tag satisfies a function tag_pattern.
//
// Supported examples:
//
//	""                    -> accepts everything
//	"gpu"                 -> requires gpu capability
//	"nvidia"              -> requires nvidia capability
//	"gpu,nvidia"          -> requires both gpu and nvidia
//	"x86"                 -> requires x86 architecture
//	"x86-tiny"            -> requires x86-tiny base tag
//	"x86-tiny/gpu"        -> requires x86-tiny and gpu
//	"*/gpu/nvidia"        -> wildcard/regex-style compatibility
//	"^x86-.*/gpu.*$"      -> explicit regex
func MachineTagMatchesRequirement(machineTag string, requirement string) bool {
	requirement = strings.TrimSpace(strings.ToLower(requirement))
	if requirement == "" {
		return true
	}

	// Comma means AND: "gpu,nvidia" requires both gpu and nvidia.
	for _, rawReq := range strings.Split(requirement, ",") {
		req := strings.TrimSpace(rawReq)
		if req == "" {
			continue
		}
		if !machineTagMatchesSingleRequirement(machineTag, req) {
			return false
		}
	}

	return true
}

func machineTagMatchesSingleRequirement(machineTag string, requirement string) bool {
	requirement = strings.TrimSpace(strings.ToLower(requirement))
	if requirement == "" {
		return true
	}

	if looksLikeRegexOrWildcard(requirement) {
		return machineTagMatchesRegexOrWildcard(machineTag, requirement)
	}

	if strings.Contains(requirement, "/") {
		return machineTagMatchesStructuredRequirement(machineTag, requirement)
	}

	return HasCapability(machineTag, requirement)
}

func machineTagMatchesStructuredRequirement(machineTag string, requirement string) bool {
	tagInfo := ParseMachineTag(machineTag)
	parts := strings.Split(requirement, "/")
	if len(parts) == 0 {
		return true
	}

	first := strings.TrimSpace(parts[0])
	if first != "" && first != "*" {
		if looksLikeBaseRequirement(first) {
			if first != tagInfo.BaseTag && first != tagInfo.Architecture && first != tagInfo.Specialization && first != tagInfo.FullTag {
				return false
			}
		} else if !HasCapability(machineTag, first) {
			return false
		}
	}

	for _, rawCapability := range parts[1:] {
		capability := strings.TrimSpace(rawCapability)
		if capability == "" || capability == "*" {
			continue
		}
		if !HasCapability(machineTag, capability) {
			return false
		}
	}

	return true
}

func looksLikeBaseRequirement(value string) bool {
	value = strings.TrimSpace(strings.ToLower(value))
	if strings.Contains(value, "-") {
		return true
	}

	switch value {
	case "x86", "x86_64", "amd64", "arm", "arm64", "aarch64":
		return true
	default:
		return false
	}
}

func looksLikeRegexOrWildcard(pattern string) bool {
	return strings.Contains(pattern, "*") || strings.HasPrefix(pattern, "^") || strings.HasSuffix(pattern, "$") || strings.Contains(pattern, ".*")
}

func machineTagMatchesRegexOrWildcard(machineTag string, pattern string) bool {
	normalizedPattern := normalizeMachineTagPattern(pattern)
	matched, err := regexp.MatchString(normalizedPattern, strings.ToLower(machineTag))
	if err != nil {
		log.Printf("[MAB] invalid machine_tag requirement pattern=%q normalized=%q error=%v\n",
			pattern,
			normalizedPattern,
			err,
		)
		return false
	}

	return matched
}

func normalizeMachineTagPattern(pattern string) string {
	if pattern == "" {
		return ""
	}
	if strings.HasPrefix(pattern, "^") || strings.HasSuffix(pattern, "$") || strings.Contains(pattern, ".*") {
		return pattern
	}

	escaped := regexp.QuoteMeta(pattern)
	escaped = strings.ReplaceAll(escaped, `\*`, ".*")
	return "^" + escaped + "$"
}
