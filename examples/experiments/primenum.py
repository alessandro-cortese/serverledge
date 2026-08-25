import time
import math


CPU_MAX_PRIME = 100000

def handler(params, context):
    max_prime = CPU_MAX_PRIME
    prime_count = 0
    iterations = 0

    start = time.time()


    for c in range(3, max_prime + 1):
        iterations = iterations + 1
        t = int(math.sqrt(c))
        is_prime = True

        for l in range(2, t + 1):
            if c % l == 0:
                is_prime = False
                break

        if is_prime:
            prime_count += 1

    duration = time.time() - start

    return {
        "function": "primes_pure",
        "max_prime_limit": max_prime,
        "primes_found": prime_count,
        "latency_seconds": duration,
        "iterations": iterations
    }