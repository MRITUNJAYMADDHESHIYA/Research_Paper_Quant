
from artha import Bot
import math
from collections import deque


class AdaptivePOVBot(Bot):

    def __init__(self):
        self.volume_history = {}
        self.return_history = {}
        self.previous_mid = {}
        # EWMA estimates
        self.ewma_volume = {}
        self.ewma_abs_return = {}

    def clamp(self, value, low, high):
        return max(low, min(high, value))
    def initialize_symbol(self, sym):
        if sym in self.volume_history:
            return

        self.volume_history[sym] = deque(maxlen=40)

        self.return_history[sym] = deque(maxlen=40)

        self.previous_mid[sym] = None

        self.ewma_volume[sym] = 0.0

        self.ewma_abs_return[sym] = 0.0

    def update_market(self, state):

        for sym in state.instruments:

            self.initialize_symbol(sym)

            volume = state.last_volume.get(sym, 0.0)

            if volume is None:
                volume = 0.0

            if not math.isfinite(volume):
                volume = 0.0

            volume = max(volume, 0.0)

            if volume > 0:

                self.volume_history[sym].append(volume)

                if self.ewma_volume[sym] <= 0:

                    self.ewma_volume[sym] = volume

                else:

                    self.ewma_volume[sym] = (
                        0.90 * self.ewma_volume[sym]
                        +
                        0.10 * volume
                    )

            mid = state.mid.get(sym, 0.0)

            if mid is None:
                mid = 0.0

            if not math.isfinite(mid):
                mid = 0.0

            previous = self.previous_mid[sym]

            if (
                previous is not None
                and previous > 0
                and mid > 0
            ):

                ret = (mid - previous) / previous

                # Protect against pathological values.
                ret = self.clamp(ret, -0.20, 0.20)

                self.return_history[sym].append(ret)

                abs_ret = abs(ret)

                if self.ewma_abs_return[sym] <= 0:

                    self.ewma_abs_return[sym] = abs_ret

                else:

                    self.ewma_abs_return[sym] = (
                        0.90 * self.ewma_abs_return[sym]
                        +
                        0.10 * abs_ret
                    )

            self.previous_mid[sym] = mid

    def get_market_condition(self, sym):

        volumes = list(self.volume_history[sym])

        returns = list(self.return_history[sym])


        if len(volumes) < 8:

            return {
                "volume_ratio": 1.0,
                "volatility_ratio": 1.0,
                "shock": False
            }


        recent_n = min(5, len(volumes))

        recent_volume = (
            sum(volumes[-recent_n:])
            /
            recent_n
        )

        long_volume = (
            sum(volumes)
            /
            len(volumes)
        )

        if long_volume > 0:

            volume_ratio = (
                recent_volume
                /
                long_volume
            )

        else:

            volume_ratio = 1.0

        if len(returns) >= 8:

            short_returns = returns[-5:]

            short_volatility = (
                sum(abs(x) for x in short_returns)
                /
                len(short_returns)
            )

            long_volatility = (
                sum(abs(x) for x in returns)
                /
                len(returns)
            )

            if long_volatility > 1e-12:

                volatility_ratio = (
                    short_volatility
                    /
                    long_volatility
                )

            else:

                volatility_ratio = 1.0

        else:

            volatility_ratio = 1.0


        shock = (
            len(volumes) >= 10
            and volume_ratio < 0.60
            and volatility_ratio > 1.15
        )

        return {
            "volume_ratio": volume_ratio,
            "volatility_ratio": volatility_ratio,
            "shock": shock
        }

    def completion(self, state, sym):

        mandate = state.mandate.get(sym, 0.0)

        remaining = state.remaining.get(sym, 0.0)

        quantity = abs(mandate)

        if quantity <= 0:

            return 1.0

        completed = (
            quantity
            -
            abs(remaining)
        )

        return self.clamp(
            completed / quantity,
            0.0,
            1.0
        )


    def on_bar(self, state):
        self.update_market(state)
        bars_left = max(state.bars_left, 1)

        active = []

        for sym in state.instruments:

            mandate = state.mandate.get(sym, 0.0)

            remaining = state.remaining.get(sym, 0.0)

            if mandate == 0:
                continue

            if abs(remaining) <= 1e-12:
                continue

            active.append(sym)

        if not active:

            return {}


        completion = {}

        for sym in active:

            completion[sym] = self.completion(
                state,
                sym
            )

        average_completion = (
            sum(completion.values())
            /
            len(completion)
        )

        # =========================================================
        # Schedule urgency
        # =========================================================

        elapsed_fraction = (
            state.bar
            /
            max(state.n_bars - 1, 1)
        )

        schedule_gap = (
            elapsed_fraction
            -
            average_completion
        )

        # Base urgency.
        urgency = 1.0 + 2.0 * schedule_gap

        urgency = self.clamp(
            urgency,
            0.80,
            2.00
        )

        # =========================================================
        # End-game acceleration
        # =========================================================

        if bars_left <= 100:

            urgency *= 1.10

        if bars_left <= 60:

            urgency *= 1.15

        if bars_left <= 30:

            urgency *= 1.25

        if bars_left <= 15:

            urgency *= 1.40

        # =========================================================
        # Market conditions
        # =========================================================

        conditions = {}

        shock_detected = False

        for sym in active:

            condition = self.get_market_condition(sym)

            conditions[sym] = condition

            if condition["shock"]:

                shock_detected = True

        # =========================================================
        # Shock response
        # =========================================================

        if shock_detected:

            urgency *= 1.25

        urgency = self.clamp(
            urgency,
            0.70,
            3.50
        )

        # =========================================================
        # Generate orders
        # =========================================================

        orders = {}

        for sym in active:

            remaining = state.remaining[sym]

            remaining_quantity = abs(remaining)

            if remaining_quantity <= 1e-12:

                continue

            condition = conditions[sym]

            volume_ratio = condition["volume_ratio"]

            # -----------------------------------------------------
            # TWAP base
            # -----------------------------------------------------

            twap_quantity = (
                remaining_quantity
                /
                bars_left
            )

            # -----------------------------------------------------
            # Volume adjustment
            #
            # Higher recent liquidity -> trade more.
            # Lower recent liquidity -> trade less.
            # -----------------------------------------------------

            volume_ratio = self.clamp(
                volume_ratio,
                0.60,
                1.50
            )

            volume_factor = math.sqrt(
                volume_ratio
            )

            desired = (
                twap_quantity
                *
                urgency
                *
                volume_factor
            )

            # -----------------------------------------------------
            # Synchronize legs.
            #
            # A lagging leg gets a modest boost.
            # -----------------------------------------------------

            completion_gap = (
                average_completion
                -
                completion[sym]
            )

            synchronization_factor = (
                1.0
                +
                completion_gap
            )

            synchronization_factor = self.clamp(
                synchronization_factor,
                0.85,
                1.20
            )

            desired *= synchronization_factor

            # -----------------------------------------------------
            # Participation target
            # -----------------------------------------------------

            target_participation = 0.10

            if urgency > 1.15:

                target_participation = 0.13

            if urgency > 1.40:

                target_participation = 0.16

            if urgency > 1.75:

                target_participation = 0.20

            if shock_detected:

                target_participation += 0.03

            target_participation = self.clamp(
                target_participation,
                0.08,
                0.24
            )

            # -----------------------------------------------------
            # Estimated current liquidity
            # -----------------------------------------------------

            estimated_volume = (
                self.ewma_volume.get(sym, 0.0)
            )

            if estimated_volume > 0:

                participation_quantity = (
                    estimated_volume
                    *
                    target_participation
                )

                # -------------------------------------------------
                # Minimum rate needed to finish.
                #
                # Never allow our participation constraint to make
                # finishing mathematically impossible.
                # -------------------------------------------------

                minimum_finish_rate = (
                    remaining_quantity
                    /
                    bars_left
                )

                if participation_quantity >= minimum_finish_rate:

                    desired = min(
                        desired,
                        participation_quantity
                    )

                else:

                    desired = minimum_finish_rate

            # -----------------------------------------------------
            # Never overfill.
            # -----------------------------------------------------

            desired = min(
                desired,
                remaining_quantity
            )

            if not math.isfinite(desired):

                continue

            # -----------------------------------------------------
            # Restore correct direction.
            # -----------------------------------------------------

            if remaining > 0:

                orders[sym] = desired

            else:

                orders[sym] = -desired

        # =========================================================
        # Emergency schedule protection
        # =========================================================

        for sym in active:

            remaining = state.remaining[sym]

            mandate = state.mandate[sym]

            if abs(remaining) <= 1e-12:

                continue

            if abs(mandate) <= 1e-12:

                continue

            remaining_fraction = (
                abs(remaining)
                /
                abs(mandate)
            )

            time_fraction = (
                bars_left
                /
                state.n_bars
            )

            if time_fraction > 0:

                pressure = (
                    remaining_fraction
                    /
                    time_fraction
                )

            else:

                pressure = 999.0

            # -----------------------------------------------------
            # We are badly behind schedule.
            # -----------------------------------------------------

            if pressure > 1.80:

                current = abs(
                    orders.get(sym, 0.0)
                )

                multiplier = self.clamp(
                    pressure / 1.80,
                    1.0,
                    2.50
                )

                desired = current * multiplier

                desired = min(
                    desired,
                    abs(remaining)
                )

                if remaining > 0:

                    orders[sym] = desired

                else:

                    orders[sym] = -desired

        # =========================================================
        # Final validation
        # =========================================================

        clean_orders = {}

        for sym, order in orders.items():

            mandate = state.mandate[sym]

            remaining = state.remaining[sym]

            # -----------------------------------------------------
            # Finite
            # -----------------------------------------------------

            if not math.isfinite(order):

                continue

            # -----------------------------------------------------
            # Correct side
            # -----------------------------------------------------

            if mandate > 0:

                order = max(order, 0.0)

            elif mandate < 0:

                order = min(order, 0.0)

            else:

                continue

            # -----------------------------------------------------
            # Never overfill
            # -----------------------------------------------------

            if abs(order) > abs(remaining):

                if remaining > 0:

                    order = abs(remaining)

                else:

                    order = -abs(remaining)

            if abs(order) > 1e-12:

                clean_orders[sym] = order

        return clean_orders


