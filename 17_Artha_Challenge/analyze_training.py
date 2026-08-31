"""
Artha Training Data Research

Run:

    python analyze_training.py

This script analyzes the 400 training sessions and creates
a historical volume profile that can later be embedded
into the trading bot.
"""

from artha.paths import load

import statistics


# ================================================================
# Configuration
# ================================================================

N_BARS = 320

INSTRUMENTS = [
    "ASHVAM",
    "BRIHAT",
    "CHAKRA",
]


# ================================================================
# Utility functions
# ================================================================

def mean(values):

    if not values:

        return 0.0

    return sum(values) / len(values)


def median(values):

    if not values:

        return 0.0

    return statistics.median(values)


def normalize(values):

    total = sum(values)

    if total <= 0:

        return [0.0 for _ in values]

    return [
        x / total
        for x in values
    ]


# ================================================================
# Load training data
# ================================================================

sessions = list(
    load("train")
)


print("=" * 80)
print("ARTHA VOLUME PROFILE RESEARCH")
print("=" * 80)

print()

print(
    "Training sessions:",
    len(sessions)
)


# ================================================================
# Shock information
# ================================================================

normal_sessions = []

shock_sessions = []

for session in sessions:

    if session.shock_at is None:

        normal_sessions.append(session)

    else:

        shock_sessions.append(session)


print(
    "Normal sessions:",
    len(normal_sessions)
)

print(
    "Shock sessions:",
    len(shock_sessions)
)

print(
    "Shock probability:",
    round(
        100.0
        * len(shock_sessions)
        / len(sessions),
        2
    ),
    "%"
)

print()


# ================================================================
# Shock locations
# ================================================================

shock_locations = [
    s.shock_at
    for s in shock_sessions
]

if shock_locations:

    print("=" * 80)
    print("SHOCK LOCATION")
    print("=" * 80)

    print(
        "Mean:",
        round(mean(shock_locations), 2)
    )

    print(
        "Median:",
        round(median(shock_locations), 2)
    )

    print(
        "Minimum:",
        min(shock_locations)
    )

    print(
        "Maximum:",
        max(shock_locations)
    )

    print()


# ================================================================
# Build normalized volume profiles
# ================================================================

profiles = {}

normal_profiles = {}

shock_profiles = {}


for sym in INSTRUMENTS:

    profiles[sym] = []

    normal_profiles[sym] = []

    shock_profiles[sym] = []


    for session in sessions:

        volumes = list(
            session.volume[sym]
        )

        if len(volumes) != N_BARS:

            continue

        profile = normalize(volumes)

        profiles[sym].append(profile)


        if session.shock_at is None:

            normal_profiles[sym].append(profile)

        else:

            shock_profiles[sym].append(profile)


# ================================================================
# Calculate average profile
# ================================================================

average_profiles = {}

median_profiles = {}

normal_average_profiles = {}

shock_average_profiles = {}


for sym in INSTRUMENTS:

    average_profiles[sym] = []

    median_profiles[sym] = []

    normal_average_profiles[sym] = []

    shock_average_profiles[sym] = []


    for bar in range(N_BARS):

        values = [
            profile[bar]
            for profile in profiles[sym]
        ]

        normal_values = [
            profile[bar]
            for profile in normal_profiles[sym]
        ]

        shock_values = [
            profile[bar]
            for profile in shock_profiles[sym]
        ]


        average_profiles[sym].append(
            mean(values)
        )

        median_profiles[sym].append(
            median(values)
        )

        normal_average_profiles[sym].append(
            mean(normal_values)
        )

        shock_average_profiles[sym].append(
            mean(shock_values)
        )


# ================================================================
# Print detailed profile
# ================================================================

for sym in INSTRUMENTS:

    print("=" * 80)
    print("INSTRUMENT:", sym)
    print("=" * 80)

    print()

    print(
        "{:>6} {:>12} {:>12} {:>12} {:>12}".format(
            "BAR",
            "AVG",
            "MEDIAN",
            "NORMAL",
            "SHOCK"
        )
    )

    print("-" * 60)


    for bar in range(
        0,
        N_BARS,
        10
    ):

        print(
            "{:>6} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}".format(
                bar,
                average_profiles[sym][bar],
                median_profiles[sym][bar],
                normal_average_profiles[sym][bar],
                shock_average_profiles[sym][bar],
            )
        )

    print()


# ================================================================
# Cumulative volume profile
# ================================================================

print("=" * 80)
print("CUMULATIVE VOLUME PROFILE")
print("=" * 80)


cumulative_profiles = {}


for sym in INSTRUMENTS:

    cumulative_profiles[sym] = []

    running = 0.0

    for value in average_profiles[sym]:

        running += value

        cumulative_profiles[sym].append(
            running
        )


for sym in INSTRUMENTS:

    print()

    print("Instrument:", sym)

    print()

    print(
        "{:>6} {:>15}".format(
            "BAR",
            "CUMULATIVE"
        )
    )

    print("-" * 30)


    for bar in range(
        0,
        N_BARS,
        10
    ):

        print(
            "{:>6} {:>15.6f}".format(
                bar,
                cumulative_profiles[sym][bar]
            )
        )


# ================================================================
# VWAP execution schedule
# ================================================================

print()
print("=" * 80)
print("RECOMMENDED HISTORICAL VWAP EXECUTION")
print("=" * 80)


# We use the average profile of ASHVAM.
#
# ASHVAM / BRIHAT / CHAKRA profiles are extremely similar
# according to the training data.


reference_profile = average_profiles["ASHVAM"]


print()

print(
    "Bar    Target cumulative execution"
)

print("-" * 50)


for bar in range(
    0,
    N_BARS,
    10
):

    target = cumulative_profiles["ASHVAM"][bar]

    print(
        "{:>3}    {:.4f} ({:.2f}%)".format(
            bar,
            target,
            target * 100.0
        )
    )


# ================================================================
# Generate Python constant
# ================================================================

print()
print("=" * 80)
print("PYTHON PROFILE CONSTANT")
print("=" * 80)

print()

print("VOLUME_PROFILE = [")

for i, value in enumerate(reference_profile):

    if i % 5 == 0:

        print("    ", end="")

    print(
        "{:.8f},".format(value),
        end=" "
    )

    if i % 5 == 4:

        print()

if len(reference_profile) % 5 != 0:

    print()

print("]")


# ================================================================
# Completion checkpoints
# ================================================================

print()
print("=" * 80)
print("EXECUTION CHECKPOINTS")
print("=" * 80)

checkpoints = [
    20,
    40,
    60,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    220,
    240,
    260,
    280,
    300,
    319,
]


for bar in checkpoints:

    target = cumulative_profiles["ASHVAM"][bar]

    print(
        "Bar {:3d} -> {:.2f}% executed".format(
            bar,
            target * 100.0
        )
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)