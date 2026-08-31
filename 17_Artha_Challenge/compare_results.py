"""
Simple score comparison helper.

Edit the numbers below after running:

    artha run starter_twap.py
    artha run adaptive_pov.py
"""


# ================================================================
# ENTER YOUR RESULTS HERE
# ================================================================

twap = {
    "impact": 0.0,
    "crowding": 0.0,
    "exposure": 0.0,
    "carry": 0.0,
    "penalty": 0.0,
}


adaptive = {
    "impact": 0.0,
    "crowding": 0.0,
    "exposure": 0.0,
    "carry": 0.0,
    "penalty": 0.0,
}


def total(result):

    return (
        result["impact"]
        +
        result["crowding"]
        +
        result["exposure"]
        +
        result["carry"]
        +
        result["penalty"]
    )


twap_score = total(twap)

adaptive_score = total(adaptive)


print("=" * 60)

print("TWAP SCORE     :", round(twap_score, 4))

print("ADAPTIVE SCORE :", round(adaptive_score, 4))

print("=" * 60)


if twap_score > 0:

    improvement = (
        (twap_score - adaptive_score)
        /
        twap_score
        *
        100.0
    )

    print(
        "Improvement:",
        round(improvement, 2),
        "%"
    )


print()

print("Component comparison")

print("-" * 60)

for key in twap:

    print(
        "{:<12} TWAP={:>10.3f}   Adaptive={:>10.3f}".format(
            key,
            twap[key],
            adaptive[key]
        )
    )
    