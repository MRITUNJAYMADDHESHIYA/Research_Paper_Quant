from artha import Bot
import math
import statistics


ENGINE_PARTICIPATION_CAP = 0.25

def safe_number(x):
    try:
        x = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return x


def positive_mean(values):
    clean = []
    for x in values:
        x = safe_number(x)
        if x > 0.0:
            clean.append(x)
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def robust_location(values):
    clean = []
    for x in values:
        x = safe_number(x)
        if x > 0.0:
            clean.append(x)

    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]

    return statistics.median(clean)


def rms(values):
    clean = []

    for x in values:
        x = safe_number(x)
        if math.isfinite(x):
            clean.append(x)

    if not clean:
        return 0.0

    return math.sqrt(sum(x * x for x in clean) / len(clean))


def geometric_blend(a, b):
    a = max(safe_number(a), 0.0)
    b = max(safe_number(b), 0.0)

    if a <= 0.0:
        return b
    if b <= 0.0:
        return a

    return math.sqrt(a * b)


def clamp_rule_based(value, upper):
    if value < 0.0:
        return 0.0

    if value > upper:
        return upper

    return value


class MyBot(Bot):
    def __init__(self):
        self.previous_mid   = {}
        self.return_history = {}

    def update_returns(self, state):
        for sym in state.instruments:
            mid = safe_number(state.mid.get(sym, 0.0))

            if sym not in self.return_history:
                self.return_history[sym] = []

            if sym in self.previous_mid:
                previous = self.previous_mid[sym]
                if previous > 0.0 and mid > 0.0:
                    ret = (mid / previous - 1.0)

                    if math.isfinite(ret):
                        self.return_history[sym].append(ret)

            if mid > 0.0:
                self.previous_mid[sym] = mid

  
    def recent_window_length(self, n):
        if n <= 1:
            return n
        return max(1, int(math.sqrt(n)))

   ########### Liquidity ##########################
    def estimate_liquidity(self, state, sym):
        history = list(state.volume_history.get(sym, []))
        clean = []

        for v in history:
            v = safe_number(v)
            if v > 0.0:
                clean.append(v)

        last_volume = safe_number(state.last_volume.get(sym, 0.0))
        if not clean:
            if last_volume > 0.0:
                return last_volume, 1.0

            return 0.0, 1.0

        n = len(clean)
        recent_n = self.recent_window_length(n)
        recent = clean[-recent_n:]
        full_level = robust_location(clean)
        recent_level = robust_location(recent)
        expected_volume = geometric_blend(full_level, recent_level)

        if expected_volume <= 0.0:
            expected_volume = positive_mean(clean)
        if expected_volume <= 0.0:
            return 0.0, 1.0

        if last_volume > 0.0:
            surprise = (last_volume/expected_volume)
        else:
            surprise = 1.0

        if not math.isfinite(surprise) or surprise <= 0.0:
            surprise = 1.0

        return expected_volume, surprise

    ######################### Continuous regime factor
    def regime_factor(self, state, sym):
        volumes = list(state.volume_history.get(sym, []))
        clean_volumes = []
        for v in volumes:
            v = safe_number(v)
            if v > 0.0:
                clean_volumes.append(v)

        returns = self.return_history.get(sym,[])

        ############ Volume Regime ##############
        volume_ratio = 1.0
        if clean_volumes:
            n = len(clean_volumes)
            recent_n = self.recent_window_length(n)
            recent_volume = robust_location(clean_volumes[-recent_n:])
            all_volume = robust_location(clean_volumes)

            if (recent_volume > 0.0 and all_volume > 0.0):
                volume_ratio = (recent_volume / all_volume)

        ################## Volatility regime ##############
        volatility_ratio = 1.0
        if returns:
            n_ret = len(returns)
            recent_n_ret = self.recent_window_length(n_ret)
            recent_volatility = rms(returns[-recent_n_ret:])
            all_volatility = rms(returns)

            if (recent_volatility > 0.0 and all_volatility > 0.0):
                volatility_ratio = (recent_volatility / all_volatility)

        # --------------------------------------------------------------
        # Continuous liquidity quality.
        #
        # Low volume and high volatility reduce the amount of future
        # liquidity we trust. There is no binary shock threshold.
        # --------------------------------------------------------------

        denominator = max(volatility_ratio, 1.0)
        quality = ( 1.0 /denominator)
        if not math.isfinite(quality) or quality <= 0.0:
            quality = 1.0

        return quality

   ######### Completion state ##############
    def completion_fraction(self, state, sym):
        mandate = safe_number(state.mandate.get(sym, 0.0))
        remaining = safe_number(state.remaining.get(sym, 0.0))
        total = abs(mandate)

        if total <= 0.0:
            return 1.0

        done = (total - abs(remaining))
        return clamp_rule_based( done / total, 1.0)

   ############## Bar ################
    def on_bar(self, state):
        self.update_returns(state)
        bars_left = max(int(state.bars_left), 1)
        active = []

        for sym in state.instruments:
            mandate = safe_number(state.mandate.get(sym, 0.0))
            remaining = safe_number(state.remaining.get(sym, 0.0))
            if abs(mandate) <= 0.0:
                continue

            if abs(remaining) <= 0.0:
                continue

            active.append(sym)

        if not active:
            return {}

        ################ Backtest ###################
        completion = {}
        for sym in active:
            completion[sym] = self.completion_fraction(state, sym)

        basket_completion = (sum(completion.values())/len(completion))
        orders = {}

    
        for sym in active:
            mandate = safe_number(state.mandate.get(sym, 0.0))
            remaining = safe_number(state.remaining.get(sym, 0.0))
            remaining_abs = abs(remaining)

            required_rate = (remaining_abs/bars_left)

            expected_volume, volume_surprise = (self.estimate_liquidity(state, sym))
            regime_quality = self.regime_factor(state,sym)


            expected_bar_capacity        = (ENGINE_PARTICIPATION_CAP* expected_volume* regime_quality)
            estimated_remaining_capacity = (expected_bar_capacity* bars_left)

            if estimated_remaining_capacity > 0.0:
                feasibility_pressure = (remaining_abs/ estimated_remaining_capacity)
            else:
                feasibility_pressure = 1.0
            if (not math.isfinite(feasibility_pressure) or feasibility_pressure <= 0.0):  feasibility_pressure = 1.0

            
            liquidity_opportunity = math.sqrt(max(volume_surprise, 0.0))
            schedule_progress     = (1.0 - bars_left / max(state.n_bars, 1))
            total_qty             = abs(mandate)

            if total_qty > 0.0:
                inventory_progress = (1.0 - remaining_abs / total_qty)
            else:
                inventory_progress = 1.0


            schedule_gap = (schedule_progress- inventory_progress)
            urgency      = math.exp(min(max(schedule_gap, -2.0), 2.0))
            sync_gap     = (basket_completion - completion[sym])
            sync_factor  = math.exp(min(max(sync_gap, -2.0), 2.0))
            adaptive_multiplier = math.sqrt(urgency  * sync_factor)


            if (not math.isfinite(adaptive_multiplier) or adaptive_multiplier <= 0.0):
                adaptive_multiplier = 1.0

            desired         = required_rate * adaptive_multiplier
            future_capacity = (expected_bar_capacity*max(bars_left - 1, 0))
            shortfall_risk  = max(0.0, remaining_abs - future_capacity)
            must_trade_now  = math.sqrt(shortfall_risk * required_rate) if shortfall_risk > 0.0 else 0.0
            desired         = max(desired, must_trade_now)
            last_volume     = safe_number(state.last_volume.get(sym, 0.0))

            if last_volume > 0.0:
                proxy_capacity = (ENGINE_PARTICIPATION_CAP * last_volume)
                desired = min(desired, max(proxy_capacity, must_trade_now))
            
            if bars_left == 1:
                desired = remaining_abs

            desired = min(desired,remaining_abs)

            if not math.isfinite(desired):
                continue

            if desired <= 0.0:
                continue

            if remaining > 0.0:
                orders[sym] = desired
            else:
                orders[sym] = -desired

        return orders
