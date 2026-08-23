package mab

// In-flight bookkeeping shared by the two policies.
//
// UCB1 and LinUCB keep different per-arm state — ArmStats holds counts and
// rewards, LinUCBArmState holds the A matrix and the b vector — but they track
// pending selections in exactly the same way: an int64 counter on the arm, an
// int64 total on the policy, and two log events.
//
// The two policies previously carried a private copy each of this logic. Since
// the counters are decremented on a path that runs under lock for every single
// invocation, keeping two copies meant that a correction could be applied to
// one and forgotten on the other. The helpers below hold the single
// implementation; the policy methods keep the map lookup, which is the only
// part that genuinely differs.

// startInFlightSelection records that one selection of arm is pending.
//
// Both counters are incremented together so that the per-arm value and the
// policy total can never drift.
func startInFlightSelection(policy string, functionName string, arm string, armInFlight *int64, totalInFlight *int64) {
	*armInFlight++
	*totalInFlight++

	logMABInFlightChanged(
		policy,
		functionName,
		arm,
		"started",
		*armInFlight,
		*totalInFlight,
	)
}

// completeInFlightSelection resolves one pending selection of arm, reporting
// whether there was one to resolve.
//
// A resolution without a matching pending selection is ignored rather than
// treated as an error: it can legitimately happen when feedback arrives for a
// selection the policy has already discarded. The total is guarded separately
// because it must never go negative even if the per-arm counter and the total
// were to disagree.
func completeInFlightSelection(policy string, functionName string, arm string, armInFlight *int64, totalInFlight *int64) bool {
	if *armInFlight <= 0 {
		logMABInFlightIgnored(
			policy,
			functionName,
			arm,
			"no_pending_selection",
		)

		return false
	}

	*armInFlight--
	if *totalInFlight > 0 {
		*totalInFlight--
	}

	logMABInFlightChanged(
		policy,
		functionName,
		arm,
		"resolved",
		*armInFlight,
		*totalInFlight,
	)

	return true
}
