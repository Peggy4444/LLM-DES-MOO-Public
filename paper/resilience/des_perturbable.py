"""
Parameterized wrapper around the Case A DES for resilience analysis.

Preserves the exact simulation logic from results/initial_model.py so that
KPIs are numerically comparable to the reported baseline. Exposes two
extension points on top of the fixed structure:

    1. `buffer_caps`  -- one capacity per named DelayBuffer.
    2. `perturb`      -- per-machine overrides:
                         { machine_name: { "mttr_mult": float,
                                           "availability_delta_pp": float,
                                           "process_time_mult": float } }

Returns a dict with throughput (parts/h), average WIP (parts), and
specific energy consumption SEC (kWh/part).
"""

import random
import statistics

import simpy


SIM_TIME = 691_200          # 8 days
WARMUP_SECONDS = 86_400     # 1 day


def production_wait_time(now):
    SEC_PER_DAY = 86_400
    day = int((now // SEC_PER_DAY) % 7)
    time_of_day = now % SEC_PER_DAY
    if day == 4:
        stop_start = 17 * 3600
        stop_end = SEC_PER_DAY + 7 * 3600
        if time_of_day >= stop_start:
            return stop_end - time_of_day
        return 0.0
    if day == 5:
        if time_of_day < 7 * 3600:
            return 7 * 3600 - time_of_day
        stop_start = 17 * 3600
        stop_end = SEC_PER_DAY + 7 * 3600
        if time_of_day >= stop_start:
            return stop_end - time_of_day
        return 0.0
    if day == 6:
        if time_of_day < 7 * 3600:
            return 7 * 3600 - time_of_day
        return 0.0
    return 0.0


def _has_free_capacity(buf):
    return (
        getattr(buf, "free_capacity", None) and buf.free_capacity() > 0
        or len(buf.items) < buf.capacity
    )


def splitter(env, input_store, out1, out2):
    toggle = 0
    while True:
        part = yield input_store.get()
        first, second = (out1, out2) if toggle == 0 else (out2, out1)
        if _has_free_capacity(first):
            yield first.put(part)
            toggle ^= 1
        else:
            yield second.put(part)


def forwarder(env, src, dst):
    while True:
        part = yield src.get()
        yield dst.put(part)


def merger(env, a, b, out):
    env.process(forwarder(env, a, out))
    env.process(forwarder(env, b, out))


def reset_machine_stats(m):
    m.working_time = 0
    m.failed_time_total = 0
    m.wait_input_time = 0
    m.blocked_time = 0
    m.processed_count = 0
    m.window_wait_time = 0


class DelayBuffer:
    def __init__(self, env, cap, delay):
        self.env = env
        self.delay = delay
        self.cap = cap
        self.store = simpy.Store(env, capacity=cap)
        self.tokens = simpy.Container(env, init=cap, capacity=cap)
        self._in_transit = 0

    def put(self, part):
        return self.env.process(self._delayed_put(part))

    def get(self):
        return self.env.process(self._get_and_release())

    @property
    def items(self):
        return self.store.items

    @property
    def capacity(self):
        return self.store.capacity

    def in_transit_count(self):
        return self._in_transit

    def free_capacity(self):
        return int(self.tokens.level)

    def _delayed_put(self, part):
        yield self.tokens.get(1)
        self._in_transit += 1
        try:
            yield self.env.timeout(self.delay)
            yield self.store.put(part)
        finally:
            self._in_transit -= 1

    def _get_and_release(self):
        part = yield self.store.get()
        yield self.tokens.put(1)
        return part


class Machine:
    def __init__(
        self, env, name, input_buffer, output_buffer, process_time,
        availability, mttr, working_power, waiting_power,
        defect_rate=None, defect_sink=None, capacity=1,
    ):
        self.env = env
        self.name = name
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.process_time = process_time
        self.availability = availability
        self.mttr = mttr
        self.defect_rate = defect_rate
        self.defect_sink = defect_sink
        self.working_power = working_power
        self.waiting_power = waiting_power
        self.resource = simpy.Resource(env, capacity=capacity)
        self.is_up = True

        self.working_time = 0
        self.failed_time_total = 0
        self.wait_input_time = 0
        self.blocked_time = 0
        self.active_count = 0
        self.processed_count = 0
        self.window_wait_time = 0

        if availability < 100:
            avail_frac = availability / 100.0
            self.mtbf = mttr * (avail_frac / (1 - avail_frac))
            env.process(self._breakdown_cycle())
        else:
            self.mtbf = float("inf")
        for _ in range(capacity):
            env.process(self.run())

    def _breakdown_cycle(self):
        while True:
            t_up = random.expovariate(1.0 / self.mtbf)
            yield self.env.timeout(t_up)
            self.is_up = False
            t_repair = random.expovariate(1.0 / self.mttr)
            yield self.env.timeout(t_repair)
            self.failed_time_total += t_repair
            self.is_up = True

    def run(self):
        while True:
            with self.resource.request() as req:
                yield req
                part = None
                while part is None:
                    if self.is_up and len(self.input_buffer.items):
                        part = yield self.input_buffer.get()
                    else:
                        if self.is_up:
                            self.wait_input_time += 1
                        yield self.env.timeout(1)

                self.processed_count += 1
                self.active_count += 1

                w = production_wait_time(self.env.now)
                self.window_wait_time += w
                if w:
                    yield self.env.timeout(w)

                pt = self.process_time() if callable(self.process_time) else self.process_time
                remaining = pt
                while remaining > 0:
                    if not self.is_up:
                        yield self.env.timeout(1)
                    else:
                        yield self.env.timeout(1)
                        self.working_time += 1
                        remaining -= 1

            start_block = self.env.now

            if self.defect_rate is not None and self.defect_sink is not None:
                if random.random() < self.defect_rate:
                    part["defect"] = 1
                    yield self.defect_sink.put(part)
                else:
                    part["defect"] = 0
                    yield self.output_buffer.put(part)
            else:
                yield self.output_buffer.put(part)

            self.blocked_time += (self.env.now - start_block)
            self.active_count -= 1

    def waiting_energy_consumption(self):
        return self.waiting_power * (
            self.wait_input_time + self.failed_time_total
            + self.blocked_time + self.window_wait_time
        )

    def working_energy_consumption(self):
        return self.working_power * self.working_time


def part_generator(env, output_buffer):
    part_id = 0
    while True:
        part = {"id": part_id}
        yield output_buffer.put(part)
        part_id += 1
        yield env.timeout(1)


def kwh_per_sec(x):
    return x / 3600.0


# ---------------------------------------------------------------------------
# Baseline machine parameters (matches results/initial_model.py exactly).
# ---------------------------------------------------------------------------
BASELINE_MACHINES = {
    "Loading robot":        dict(process_time=12.0,  availability=90.49, mttr=68.0,
                                 working_power=kwh_per_sec(0.72),  waiting_power=kwh_per_sec(0.25)),
    "Conveyor belt":        dict(process_time=6.0,   availability=100.0, mttr=1.0,
                                 working_power=kwh_per_sec(0.0),   waiting_power=kwh_per_sec(0.0)),
    "Washing machine":      dict(process_time=14.0,  availability=80.89, mttr=269.0,
                                 working_power=kwh_per_sec(35.24), waiting_power=kwh_per_sec(4.28)),
    "Hantering cell":       dict(process_time=25.0,  availability=97.79, mttr=74.0,
                                 working_power=kwh_per_sec(0.74),  waiting_power=kwh_per_sec(0.50)),
    "Presses cell 1":       dict(process_time=175.0, availability=87.79, mttr=73.0,
                                 working_power=kwh_per_sec(1.28),  waiting_power=kwh_per_sec(1.25)),
    "Presses cell 2":       dict(process_time=176.0, availability=87.69, mttr=74.0,
                                 working_power=kwh_per_sec(1.27),  waiting_power=kwh_per_sec(1.25)),
    "Quality station cell": dict(process_time=41.0,  availability=85.87, mttr=66.0,
                                 working_power=kwh_per_sec(0.84),  waiting_power=kwh_per_sec(0.58),
                                 defect_rate=0.089),
}

MACHINE_ORDER = [
    "Loading robot", "Conveyor belt", "Washing machine", "Hantering cell",
    "Presses cell 1", "Presses cell 2", "Quality station cell",
]

# Default buffer caps used in the un-optimized baseline model.
DEFAULT_BUFFER_CAPS = {
    "PostLoadingBuffer":   2,
    "PostConveyorBuffer":  2,
    "PostWashingBuffer":   2,
    "PrePress1Buffer":     3,
    "PrePress2Buffer":     3,
    "PostPress12Buffer":   3,
}

BUFFER_DELAY = {
    "PostLoadingBuffer":   10,
    "PostConveyorBuffer":  10,
    "PostWashingBuffer":   10,
    "PrePress1Buffer":     32,
    "PrePress2Buffer":     32,
    "PostPress12Buffer":   32,
}


def _apply_perturbation(params, pert):
    """Return a new params dict with perturbations applied. Non-destructive."""
    if not pert:
        return dict(params)
    p = dict(params)
    if "mttr_mult" in pert:
        p["mttr"] = p["mttr"] * pert["mttr_mult"]
    if "availability_delta_pp" in pert:
        new_av = p["availability"] + pert["availability_delta_pp"]
        # Clamp: avoid MTBF blowup and negative availability.
        p["availability"] = max(min(new_av, 99.9), 1.0)
    if "process_time_mult" in pert:
        p["process_time"] = p["process_time"] * pert["process_time_mult"]
    return p


def run_perturbed_simulation(
    seed,
    buffer_caps=None,
    perturb=None,
    warmup=WARMUP_SECONDS,
    measure_until=SIM_TIME,
):
    """
    Run a single replication of the Case A DES with optional perturbations.

    Parameters
    ----------
    seed : int
    buffer_caps : dict[str, int] | None
        Override capacity for one or more DelayBuffers. Missing keys fall back
        to DEFAULT_BUFFER_CAPS.
    perturb : dict[str, dict] | None
        Per-machine parameter perturbations. See module docstring.

    Returns
    -------
    dict with keys 'throughput', 'wip', 'sec', 'produced', 'machine_energy'.
    """
    random.seed(seed)
    env = simpy.Environment()

    caps = dict(DEFAULT_BUFFER_CAPS)
    if buffer_caps:
        caps.update(buffer_caps)

    PostLoadingBuffer  = DelayBuffer(env, caps["PostLoadingBuffer"],  BUFFER_DELAY["PostLoadingBuffer"])
    PostConveyorBuffer = DelayBuffer(env, caps["PostConveyorBuffer"], BUFFER_DELAY["PostConveyorBuffer"])
    PostWashingBuffer  = DelayBuffer(env, caps["PostWashingBuffer"],  BUFFER_DELAY["PostWashingBuffer"])
    PrePress1Buffer    = DelayBuffer(env, caps["PrePress1Buffer"],    BUFFER_DELAY["PrePress1Buffer"])
    PrePress2Buffer    = DelayBuffer(env, caps["PrePress2Buffer"],    BUFFER_DELAY["PrePress2Buffer"])
    PostPress12Buffer  = DelayBuffer(env, caps["PostPress12Buffer"],  BUFFER_DELAY["PostPress12Buffer"])

    raw_input = simpy.Store(env, capacity=1000)
    sink = simpy.Store(env, capacity=100_000)
    defects = simpy.Store(env, capacity=100_000)

    helper_to_press1 = simpy.Store(env, capacity=3)
    branch1_out = simpy.Store(env, capacity=3)
    branch2_out = simpy.Store(env, capacity=3)

    # Build all machines with perturbations applied.
    def build_machine(name, in_buf, out_buf, defect_sink=None):
        base = BASELINE_MACHINES[name]
        p = _apply_perturbation(base, (perturb or {}).get(name))
        kw = dict(
            process_time=p["process_time"],
            availability=p["availability"],
            mttr=p["mttr"],
            working_power=p["working_power"],
            waiting_power=p["waiting_power"],
        )
        if "defect_rate" in p:
            kw["defect_rate"] = p["defect_rate"]
            kw["defect_sink"] = defect_sink
        return Machine(env, name, input_buffer=in_buf, output_buffer=out_buf, **kw)

    Loading_robot    = build_machine("Loading robot",    raw_input,          PostLoadingBuffer)
    Conveyor_belt    = build_machine("Conveyor belt",    PostLoadingBuffer,  PostConveyorBuffer)
    Washing_machine  = build_machine("Washing machine",  PostConveyorBuffer, PostWashingBuffer)
    Hantering_cell   = build_machine("Hantering cell",   PostWashingBuffer,  helper_to_press1)

    env.process(splitter(env, helper_to_press1, PrePress1Buffer, PrePress2Buffer))

    Presses_cell_1 = build_machine("Presses cell 1", PrePress1Buffer, branch1_out)
    Presses_cell_2 = build_machine("Presses cell 2", PrePress2Buffer, branch2_out)

    merger(env, branch1_out, branch2_out, PostPress12Buffer)

    Quality_station_cell = build_machine(
        "Quality station cell", PostPress12Buffer, sink, defect_sink=defects,
    )

    machines_list = [
        Loading_robot, Conveyor_belt, Washing_machine, Hantering_cell,
        Presses_cell_1, Presses_cell_2, Quality_station_cell,
    ]

    env.process(part_generator(env, raw_input))
    env.run(until=warmup)
    for m in machines_list:
        reset_machine_stats(m)

    produced_count_before = len(sink.items)

    wip_samples = []
    delay_buffers = [
        PostLoadingBuffer, PostConveyorBuffer, PostWashingBuffer,
        PrePress1Buffer, PrePress2Buffer, PostPress12Buffer,
    ]

    def sample_wip(env):
        while True:
            ready = sum(len(b.items) for b in delay_buffers)
            in_transit = sum(b.in_transit_count() for b in delay_buffers)
            in_machines = sum(m.active_count for m in machines_list)
            wip_samples.append(ready + in_transit + in_machines)
            yield env.timeout(60)

    env.process(sample_wip(env))
    env.run(until=measure_until)

    total_produced = len(sink.items) - produced_count_before
    hours = (measure_until - warmup) / 3600.0
    throughput = (total_produced / hours) if hours > 0 else 0.0
    avg_wip = statistics.mean(wip_samples) if wip_samples else 0.0

    machine_energy = {}
    for m in machines_list:
        we = m.working_energy_consumption()
        wa = m.waiting_energy_consumption()
        machine_energy[m.name] = {
            "working_energy": we,
            "waiting_energy": wa,
            "total_energy":   we + wa,
        }
    total_energy = sum(v["total_energy"] for v in machine_energy.values())
    sec = total_energy / total_produced if total_produced > 0 else 0.0

    return {
        "throughput": throughput,
        "wip":        avg_wip,
        "sec":        sec,
        "produced":   total_produced,
        "machine_energy": machine_energy,
    }


if __name__ == "__main__":
    # Sanity check against the unperturbed baseline.
    import time
    t0 = time.time()
    res = run_perturbed_simulation(seed=11)
    print(f"[baseline seed=11 in {time.time()-t0:.2f} s]  "
          f"TP={res['throughput']:.3f} WIP={res['wip']:.3f} SEC={res['sec']:.4f}")
