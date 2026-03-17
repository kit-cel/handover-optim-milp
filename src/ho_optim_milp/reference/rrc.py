"""Radio resource control (RRC) implementation."""

from enum import auto, IntEnum
import logging
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from .core import Priority

if TYPE_CHECKING:
    from .core import ScheduledEvent, VirtualClock
    from .rrc_config import RRCConfig

DEBUG_LVL = logging.WARNING


logging.basicConfig(
    level=DEBUG_LVL,
    format="%(message)s",
)


class RrcState(IntEnum):
    """RRC States."""

    RRC_IDLE = auto()

    CONNECTED_NORMAL = auto()
    CONNECTED_HO_PREP = auto()
    CONNECTED_HO_EXEC = auto()

    RLF_RECOVERY = auto()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{__class__.__name__}.{self.name}"


Transition = tuple[RrcState, str]  # (current_state, event_name)


class ReferenceRRC:
    """RRC UE entity (simplified)."""

    rsrp: np.ndarray
    sinr: np.ndarray
    q_in: np.ndarray
    q_out: np.ndarray

    def __init__(
        self, imsi: int, clock: "VirtualClock", rrc_config: "RRCConfig"
    ) -> None:
        self.imsi = imsi
        self.clock = clock

        self.sync_signal_update_interval_ms = rrc_config.t_res_ms
        self.msr_decision_interval_ms = rrc_config.t_res_ms

        self.q_in_db_const = rrc_config.q_in_db
        self.q_out_db_const = rrc_config.q_out_db

        self.n310_const = rrc_config.n310
        self.n311_const = rrc_config.n311

        self.t310_ms = rrc_config.t310_ms

        self.t_ho_prep_ms = rrc_config.t_ho_prep_ms_simulated
        self.t_ho_exec_ms = rrc_config.t_ho_exec_ms_simulated
        self.t_rlfr_ms_simulated = rrc_config.t_rlfr_ms_simulated
        self.t_mts_ms = rrc_config.t_mts_ms

        self.a3_ttt_ms = rrc_config.ho_event_config["a3"]["ttt_ms"]
        self.a3_hyst_db = rrc_config.ho_event_config["a3"]["hys"]
        self.a3_off_db = rrc_config.ho_event_config["a3"]["offset"]

        self.enable_ttt_abort = False  # allow TTT abort on different candidate cell

        self.debug_lvl = logging.INFO

        self.state = RrcState.RRC_IDLE

        self._initialized = False

        self._sync_indicator_cnt = 0  # counts N310/N311 events

        self._pcell: int | None = None
        self._prev_pcell: int | None = None
        self._report_cell: int | None = None
        self._ho_target_cell: int | None = None

        self.flag_ho_exec_started: bool = False
        self.flag_ho_exec_success: bool = False
        self.flag_hof_detected: bool = False
        self.flag_rlf_started: bool = False

        # Handles for the timers
        self._h_t310: "ScheduledEvent | None" = None  # Priority HIGH
        self._h_ttt: "ScheduledEvent | None" = None  # Priority NORMAL
        self._h_prep: "ScheduledEvent | None" = None  # Priority NORMAL
        self._h_exec: "ScheduledEvent | None" = None  # Priority NORMAL
        self._h_rlf: "ScheduledEvent | None" = None  # Priority HIGH

        # RRC state transitions
        self._transitions: dict[Transition, Callable[[], None]] = {
            (RrcState.RRC_IDLE, "attach"): self._do_attach,
            #
            (RrcState.CONNECTED_NORMAL, "t310_expired"): self._rlf_due_to_t310,
            (RrcState.CONNECTED_NORMAL, "a3_enter"): self._start_ttt,
            (RrcState.CONNECTED_NORMAL, "a3_reset"): self._cancel_ttt,
            (RrcState.CONNECTED_NORMAL, "ttt_expired"): self._start_prep,
            #
            (RrcState.CONNECTED_HO_PREP, "t310_expired"): self._hof_due_to_t310_expiry,
            (RrcState.CONNECTED_HO_PREP, "prep_expired"): self._start_exec,
            (RrcState.CONNECTED_HO_PREP, "prep_failed"): self._hof_due_to_t310,
            #
            (RrcState.CONNECTED_HO_EXEC, "exec_expired"): self._finish_exec,
            (RrcState.CONNECTED_HO_EXEC, "exec_failed"): self._hof_due_to_oos,
            #
            (RrcState.RLF_RECOVERY, "rlf_expired"): self._recover_from_rlf,
        }

        # Flags
        self._flags = {
            "rlf_start": [0] * 0,
            "rlf_expire": [0] * 0,
            #
            "t310_start": [0] * 0,
            "t310_cancel": [0] * 0,
            "t310_expire": [0] * 0,
            #
            "ttt_start": [0] * 0,
            "ttt_cancel": [0] * 0,
            "ttt_expire": [0] * 0,
            #
            "prep_start": [0] * 0,
            "prep_cancel": [0] * 0,
            "prep_expire": [0] * 0,
            #
            "exec_start": [0] * 0,
            "exec_cancel": [0] * 0,
            "exec_expire": [0] * 0,
        }

    def _trigger(self, event_name: str) -> None:
        key = (self.state, event_name)
        if key not in self._transitions:
            raise RuntimeError(f"Invalid transition: ({self.state}, {event_name})")
        self._transitions[key]()

    ### RACH
    def initial_access(self) -> None:
        """Perform initial access procedure."""
        if self._initialized:
            raise RuntimeError("RRC UE already initialized")
        self._initialized = True
        self._trigger("attach")

    def _do_attach(self) -> None:
        """Attach procedure - simplified."""
        if self.rsrp is None:
            raise RuntimeError("RRC UE not initialized with PHY measurements")
        if self.state != RrcState.RRC_IDLE:
            raise RuntimeError("RRC UE not in IDLE state for attach")
        if self._pcell is not None:
            raise RuntimeError("RRC UE already has a primary cell")
        self._log("Attaching", logging.INFO)
        self._pcell = int(np.argmax(self.rsrp))
        self.state = RrcState.CONNECTED_NORMAL

    ### PHY updates and sync state
    def receive_phy_update(
        self,
        rsrp_dbm: np.ndarray,
        sinr_db: np.ndarray,
        q_in: np.ndarray,
        q_out: np.ndarray,
    ) -> None:
        """Receive PHY update with RSRP and SINR measurements."""
        rsrp_dbm = rsrp_dbm.flatten()
        sinr_db = sinr_db.flatten()
        n_bs = rsrp_dbm.shape[0]
        assert (
            rsrp_dbm.shape == sinr_db.shape == q_in.shape == q_out.shape and n_bs >= 1
        )

        self.rsrp = rsrp_dbm
        self.sinr = sinr_db
        self.q_in = q_in
        self.q_out = q_out

        self._ensure_flags()

        # Reset flags for this time step
        self.flag_ho_exec_started = False
        self.flag_ho_exec_success = False
        self.flag_hof_detected = False
        self.flag_rlf_started = False

    def update_sync_state(self) -> None:
        """Update sync state based on RSRP measurements."""
        if not self._initialized:
            raise RuntimeError("RRC UE not initialized with PHY measurements")

        if self._h_rlf is not None:
            return  # RLF timer running, no sync checks
        if self._h_exec is not None:
            return  # HO execution running, no sync checks

        q_in_pcell = self.q_in[self._pcell]
        q_out_pcell = self.q_out[self._pcell]
        if q_in_pcell and self._h_t310 is not None:
            self._in_sync()
        elif q_out_pcell and self._h_t310 is None:
            self._out_of_sync()
        else:
            pass  # no change

    def _in_sync(self) -> None:
        """Called when T311 fires."""
        self._sync_indicator_cnt += 1
        self._log(f"in_sync cnt={self._sync_indicator_cnt}", logging.INFO)
        if self._sync_indicator_cnt == self.n311_const:
            self._log("N311 reached maximum", logging.INFO)
            self._sync_indicator_cnt = 0
            self._stop_t310()

    def _out_of_sync(self) -> None:
        """Called when T310 fires."""
        self._sync_indicator_cnt += 1
        self._log(f"out_of_sync cnt={self._sync_indicator_cnt}", logging.INFO)
        if self._sync_indicator_cnt == self.n310_const:
            self._log("N310 reached maximum", logging.INFO)
            self._sync_indicator_cnt = 0
            self._start_t310()

    ### T310 timer
    def _start_t310(self) -> None:
        if self._h_t310 is not None:
            raise RuntimeError("T310 already running")
        self._log("T310 started", logging.INFO, flag="t310_start")
        self._h_t310 = self.clock.schedule(
            self.t310_ms,
            self._t310_expired,
            priority=Priority.HIGH,
            name="T310",
        )
        self._sync_indicator_cnt = 0

    def _stop_t310(self) -> None:
        if self._h_t310 is None:
            raise RuntimeError("T310 not running")
        self._log("T310 stopped", logging.INFO, flag="t310_cancel")
        self.clock.cancel(self._h_t310)
        self._h_t310 = None
        self._sync_indicator_cnt = 0

    def _t310_expired(self) -> None:
        self._log("T310 expired", logging.INFO, flag="t310_expire")
        self._h_t310 = None
        self._trigger("t310_expired")

    def _rlf_due_to_t310(self) -> None:
        """T310 expired → start the *RLF* timer (the official 3GPP rule)."""
        self._log("Declaring RLF due to T310 expiry", logging.INFO)
        self._cancel_ttt()
        self._start_rlf_timer()

    ### RLF timer
    def _start_rlf_timer(self) -> None:
        if self._h_rlf is not None:
            raise RuntimeError("RLF timer already running")
        self._log("RLF timer started", logging.INFO, flag="rlf_start")
        self.flag_rlf_started = True
        self._h_rlf = self.clock.schedule(
            self.t_rlfr_ms_simulated,
            self._rlf_timer_expired,
            priority=Priority.HIGH,
            name="RLF",
        )
        self.state = RrcState.RLF_RECOVERY
        self._prev_pcell = self._pcell

    def _rlf_timer_expired(self) -> None:
        self._log("RLF timer expired", logging.INFO, flag="rlf_expire")
        self._h_rlf = None
        self._trigger("rlf_expired")

    def _recover_from_rlf(self) -> None:
        """RLF timer expired → recover to CONNECTED_NORMAL."""
        self._log("Recovered from RLF", logging.INFO)
        self._pcell = None
        self.state = RrcState.RRC_IDLE
        self._trigger("attach")  # simplified for this example

    ### Measurements and TTT
    def process_measurements(self) -> None:
        """Process measurements and decide on handovers."""
        if self.state != RrcState.CONNECTED_NORMAL:
            return  # only process in CONNECTED_NORMAL
        self._measure_a3()

    def _measure_a3(self) -> None:
        if self._pcell is None:
            raise RuntimeError("No primary cell set")

        # Find best candidate cell (highest RSRP excluding PCell)
        neighbor_cell_idxs = np.delete(np.arange(len(self.rsrp)), self._pcell)
        rsrp_neighbor_cells = np.delete(self.rsrp, self._pcell)
        best_ncell = neighbor_cell_idxs[np.argmax(rsrp_neighbor_cells)]

        # Primary cell measurement (RSRP)
        mp = self.rsrp[self._pcell]
        # Reporting cell measurement (RSRP), or -inf if None
        mn = -np.inf if self._report_cell is None else self.rsrp[self._report_cell]
        # Best candidate cell measurement (RSRP)
        mc = self.rsrp[best_ncell]

        if self._report_cell is None:  # No TTT running
            if mc - self.a3_hyst_db > mp + self.a3_off_db:  # A3 entry condition
                self._report_cell = best_ncell
                self._trigger("a3_enter")  # Start TTT
        else:  # TTT running
            if mn + self.a3_hyst_db < mp + self.a3_off_db:
                # A3 exit condition
                self._trigger("a3_reset")  # Cancel TTT

            if self.enable_ttt_abort and best_ncell != self._report_cell:
                if (
                    mc - self.a3_hyst_db > mp + self.a3_off_db
                ):  # A3 entry condition for new candidate
                    # Switch to new candidate cell
                    self._log(f"Cell changed {self._report_cell} -> {best_ncell}")
                    self._trigger("a3_reset")  # Cancel TTT
                    self._report_cell = best_ncell
                    self._trigger("a3_enter")  # Re-start TTT

    ### TTT timer
    def _start_ttt(self) -> None:
        if self._h_ttt is not None:
            return
        self._log("TTT started", logging.INFO, flag="ttt_start")
        self._h_ttt = self.clock.schedule(
            self.a3_ttt_ms,
            self._ttt_expired,
            priority=Priority.NORMAL,
            name="TTT",
        )
        # print(f"t={self.clock.now} TTT started for report_cell={self._report_cell}")

    def _cancel_ttt(self) -> None:
        if self._h_ttt is None:
            return
        self._log("TTT cancelled", logging.INFO, flag="ttt_cancel")
        self.clock.cancel(self._h_ttt)
        self._h_ttt = None
        self._report_cell = None
        # print(f"t={self.clock.now} TTT cancelled, report_cell cleared")

    def _ttt_expired(self) -> None:
        self._log("TTT expired", logging.INFO, flag="ttt_expire")
        self._h_ttt = None
        self._trigger("ttt_expired")
        # print(f"t={self.clock.now} TTT expired for report_cell={self._report_cell}")

    ### Handover preparation
    def _start_prep(self) -> None:
        if self._h_prep is not None:
            raise RuntimeError("HO preparation already running")
        if self._report_cell is None:
            raise RuntimeError("No reporting cell for measurement report (HO prep)")
        self._log("HO preparation started", logging.INFO, flag="prep_start")
        self.state = RrcState.CONNECTED_HO_PREP
        self._h_prep = self.clock.schedule(
            self.t_ho_prep_ms,
            self._prep_expired,
            priority=Priority.NORMAL,
            name="HO Prep",
        )

    def _cancel_prep(self) -> None:
        if self._h_prep is None:
            raise RuntimeError("HO preparation not running")
        self._log("HO preparation cancelled", logging.INFO, flag="prep_cancel")
        self.clock.cancel(self._h_prep)
        self._h_prep = None
        self._report_cell = None
        self.state = RrcState.CONNECTED_NORMAL

    def _prep_expired(self) -> None:
        if self._h_rlf is not None:
            return  # RLF timer running, prep not relevant
        if self._report_cell is None:
            raise RuntimeError("No reporting cell for HO preparation")
        self._log("HO preparation expired", logging.INFO, flag="prep_expire")
        if self._h_t310 is not None:  # T310 is running
            self._trigger("prep_failed")  # HOF/RLF due to T310
        else:
            self._h_prep = None
            self._trigger("prep_expired")  # proceed to execution

    def _hof_due_to_t310_expiry(self) -> None:
        self._log("HOF due to T310 expiry")
        self.flag_hof_detected = True
        self._cancel_prep()
        self._start_rlf_timer()

    def _hof_due_to_t310(self) -> None:
        self._log("HOF due to T310")
        self.flag_hof_detected = True
        self._cancel_prep()
        self._stop_t310()
        self._start_rlf_timer()

    ### Handover execution
    def _start_exec(self) -> None:
        if self._h_exec is not None:
            raise RuntimeError("HO execution already running")
        if self._h_t310 is not None:
            raise RuntimeError("Cannot start HO execution while T310 is running")
        self._log("HO execution started", logging.INFO, flag="exec_start")
        self._ho_target_cell = self._report_cell
        self._report_cell = None
        self._sync_indicator_cnt = 0
        self.state = RrcState.CONNECTED_HO_EXEC
        self.flag_ho_exec_started = True
        self._h_exec = self.clock.schedule(
            self.t_ho_exec_ms,
            self._exec_expired,
            priority=Priority.NORMAL,
            name="HO Exec",
        )

    def _cancel_exec(self) -> None:
        if self._h_exec is None:
            raise RuntimeError("HO execution not running")
        self._log("HO execution cancelled", logging.INFO, flag="exec_cancel")
        self.clock.cancel(self._h_exec)
        self._h_exec = None
        self._ho_target_cell = None
        self.state = RrcState.CONNECTED_NORMAL

    def _exec_expired(self) -> None:
        if self._ho_target_cell is None:
            raise RuntimeError("No target cell for HO execution")
        if self.sinr is None:
            raise RuntimeError("RRC UE not initialized with PHY measurements")
        self._log("HO execution expired", logging.INFO, flag="exec_expire")
        if self.q_out[self._ho_target_cell]:  # poor signal on target cell
            self._trigger("exec_failed")  # HOF/RLF due to poor signal
        else:
            self._h_exec = None
            self.flag_ho_exec_success = True
            self._trigger("exec_expired")  # complete HO

    def _finish_exec(self) -> None:
        self._log("HO execution finished")
        self._prev_pcell = self._pcell
        self._pcell = self._ho_target_cell
        self._ho_target_cell = None
        self.state = RrcState.CONNECTED_NORMAL

    def _hof_due_to_oos(self) -> None:
        self._log("HOF due to poor signal on target cell")
        self._cancel_exec()
        self._start_rlf_timer()

    ### Logging helpers
    def _log(self, msg: str, lvl: int = logging.INFO, flag: str | None = None) -> None:
        """Log the current RRC state."""
        if flag is not None:
            self._set_flag(flag)
        if self.debug_lvl <= lvl:
            n_digits = len(str(self.clock.n_steps))
            if lvl == logging.DEBUG:
                logging.debug(
                    "[RRC] t=%0*d UE=%s %s %s",
                    n_digits,
                    self.clock.now,
                    self.imsi,
                    self.state,
                    msg,
                )
            elif lvl == logging.INFO:
                logging.info(
                    "[RRC] t=%0*d UE=%s %s",
                    n_digits,
                    self.clock.now,
                    self.imsi,
                    msg,
                )

    def _set_flag(self, flag_name: str, value: int = 1) -> None:
        """Set a flag at the current time."""
        if flag_name not in self._flags:
            raise ValueError(f"Unknown flag: {flag_name}")
        self._flags[flag_name][-1] = value

    def get_flags(self) -> dict[str, np.ndarray]:
        """Get flags as numpy arrays."""
        return {k: np.array(v) for k, v in self._flags.items()}

    def _ensure_flags(self) -> None:
        length = self.clock.now // self.sync_signal_update_interval_ms + 1
        for _, v in self._flags.items():
            if len(v) < length:
                v.extend([0] * (length - len(v)))

    def snapshot(self) -> dict[str, Any]:
        """Get a snapshot of the current RRC UE state including all properties."""
        return {
            "time_ms": self.clock.now,
            "imsi": self.imsi,
            "state": self.state.numerator,
            "in_sync": self.in_sync,
            "out_of_sync": self.out_of_sync,
            "connected": self.connected,
            "n310_count": self.n310_count,
            "n311_count": self.n311_count,
            "t310_remaining_ms": self.t310_remaining_ms,
            "t_rlf_remaining_ms": self.t_rlf_remaining_ms,
            "pcell": self.pcell,
            "prev_pcell": self.prev_pcell,
            "reporting_cell": self._report_cell,
            "target_cell": self.target_cell,
            "ttt_active": self.ttt_active,
            "ho_prep_ongoing": self.ho_prep_ongoing,
            "ho_exec_ongoing": self.ho_exec_ongoing,
            "t310_active": self.t310_active,
            "rlf_timer_active": self.rlfr_ongoing,
            "rsrp_dbm": self.rsrp.tolist(),
            "sinr_db": self.sinr.tolist(),
            # Flags
            **{f"flag_{k}": v[-1] for k, v in self._flags.items()},
        }

    ### Properties
    @property
    def pcell(self) -> int | None:
        """Get the current primary cell."""
        return self._pcell

    @property
    def prev_pcell(self) -> int | None:
        """Get the previous primary cell."""
        return self._prev_pcell

    @property
    def reporting_cell(self) -> int | None:
        """Get the current reporting cell."""
        return self._report_cell

    @property
    def target_cell(self) -> int | None:
        """Get the current hand-over target cell."""
        return self._ho_target_cell

    @property
    def in_sync(self) -> bool:
        """Check if UE is in sync."""
        if self._pcell is None:
            raise RuntimeError("No primary cell set")
        return self.q_in[self._pcell]

    @property
    def out_of_sync(self) -> bool:
        """Check if UE is out of sync."""
        if self._pcell is None:
            raise RuntimeError("No primary cell set")
        return self.q_out[self._pcell]

    @property
    def connected(self) -> bool:
        """Check if UE is in a connected state."""
        return self.state in {RrcState.CONNECTED_NORMAL, RrcState.CONNECTED_HO_PREP}

    @property
    def ttt_active(self) -> bool:
        """Check if TTT is active."""
        return self._h_ttt is not None

    @property
    def ho_prep_ongoing(self) -> bool:
        """Check if hand-over preparation is ongoing."""
        return self._h_prep is not None

    @property
    def ho_exec_ongoing(self) -> bool:
        """Check if hand-over execution is ongoing."""
        return self._h_exec is not None

    @property
    def t310_active(self) -> bool:
        """Check if T310 is active."""
        return self._h_t310 is not None

    @property
    def rlfr_ongoing(self) -> bool:
        """Check if RLF timer is active."""
        return self._h_rlf is not None

    @property
    def n310_count(self) -> int:
        """Get the current N310 count."""
        return self._sync_indicator_cnt if self._h_t310 is None else 0

    @property
    def n311_count(self) -> int:
        """Get the current N311 count."""
        return self._sync_indicator_cnt if self._h_t310 is not None else 0

    @property
    def t310_remaining_ms(self) -> int | None:
        """Get remaining time for T310 (None if not active)."""
        if self._h_t310 is None:
            return None
        return self._h_t310.time - self.clock.now

    @property
    def t_rlf_remaining_ms(self) -> int | None:
        """Get remaining time for RLF timer (None if not active)."""
        if self._h_rlf is None:
            return None
        return self._h_rlf.time - self.clock.now
