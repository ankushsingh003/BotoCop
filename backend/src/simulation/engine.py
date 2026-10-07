"""
Enterprise Cyber Security Simulation Engine & Realistic Threat Stream Generator.
Provides continuous, realistic traffic simulation with controllable rates and
scenario injection (Digital Arrest, Banking OTP, Boiler Room Fan-Out, Voice Deepfake,
Blocklist Short-Circuit, and Benign Calls).
"""
import time
import random
import uuid
import logging
import threading
from collections import deque
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from backend.src.orchestrator.orchestrator import handle_event

logger = logging.getLogger("simulation-engine")

BENIGN_TRANSCRIPTS = [
    "Hello, I am calling to confirm my doctor appointment scheduled for tomorrow at 10 AM.",
    "Good afternoon, this is customer care regarding your inquiry about broadband fiber plans.",
    "Hi, your courier delivery from BlueDart is arriving in 30 minutes, please confirm address.",
    "Hello, following up on the quarterly report emailed yesterday. Let me know when you can review it.",
    "Hi mom, just reached the office, will call you back during the evening lunch break.",
    "Good morning, this is Apollo Pharmacy regarding your refill order ready for pickup.",
]

DIGITAL_ARREST_TRANSCRIPTS = [
    "Namaste. Main Mumbai Police Cyber Crime Cell se Inspector Sharma bol raha hu. Aapke Aadhaar card se illegal money laundering account open hua hai. Abhi digital arrest notice issue hua hai, Supreme Court safe custody me verification deposit bhejo.",
    "Attention: CBI Special Task Force officer speaking. A parcel under your national ID containing prohibited contraband was intercepted at Delhi Airport. To stop immediate non-bailable warrant, comply with our forensic verification protocol right now.",
    "Yeh Enforcement Directorate (ED) headquarters se formal warning hai. Aapke bank khate me 45 lakh ke illegal hawala transactions detect hue hain. Turant security bond deposit verify karein ya local police dispatch hogi.",
    "Legal summons notice: Telecom Regulatory Authority and Supreme Court legal team. Your cellular SIM will be disconnected in 2 hours due to cyber intimidation complaints. Stay on the line for identity clearance.",
]

BANKING_OTP_TRANSCRIPTS = [
    "Urgent security alert from HDFC Bank Risk Department. Unauthorized transaction of Rs 84,500 detected on your credit card. Share the 6-digit OTP sent to your phone immediately to block this transaction.",
    "SBI YONO Security Alert: Your mobile banking access is suspended due to KYC non-compliance. Provide the 6-digit verification code received on SMS to reactivate your account instantly.",
    "ICICI Bank Fraud Prevention Unit: A new beneficiary addition was requested from an unknown IP address. If this was not you, press 1 and share the cancellation OTP with our automated system.",
    "Axis Bank Alert: Your debit card is temporarily blocked due to multiple failed ATM attempts. To verify cardholder identity, read back the authentication code sent to your registered mobile.",
]

VOICE_CLONE_TRANSCRIPTS = [
    "Hey Rahul, it's Vikram from executive office. We are in the middle of closing an emergency vendor contract before 5 PM. I need you to authorize an immediate RTGS transfer of 25 Lakhs. I'm boarding a flight now, execute right away.",
    "Mom, please don't panic, but I got into a car accident near Goa border. My phone is broken and police are holding my license. The towing inspector needs 45,000 rupees immediately to clear me. Please send it to his UPI number right now.",
]

BOILER_ROOM_TRANSCRIPTS = [
    "Urgent notification regarding your pre-approved loan of 10 Lakh rupees with zero interest. Confirm your account details and OTP to disburse within 15 minutes.",
    "Final reminder: Your electricity connection will be disconnected tonight at 9 PM due to unpaid bill. Click the link or give your payment OTP immediately.",
]


class SimulationEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SimulationEngine, cls).__new__(cls)
                cls._instance._init_engine()
            return cls._instance

    def _init_engine(self):
        self.is_running: bool = True
        self.interval_seconds: float = 6.0
        self.total_generated: int = 0
        self.scenario_stats: Dict[str, int] = {
            "digital_arrest": 0,
            "banking_otp": 0,
            "boiler_room": 0,
            "voice_clone": 0,
            "blocklist_trigger": 0,
            "benign_call": 0,
            "cross_channel": 0,
        }
        self.recent_events = deque(maxlen=60)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._boiler_room_caller = "+919999000888"

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "RUNNING" if self.is_running else "PAUSED",
            "interval_seconds": self.interval_seconds,
            "total_generated": self.total_generated,
            "scenario_stats": dict(self.scenario_stats),
            "recent_count": len(self.recent_events),
        }

    def set_running(self, running: bool):
        self.is_running = running
        logger.info(f"Simulation engine state updated: is_running={self.is_running}")

    def set_interval(self, seconds: float):
        self.interval_seconds = max(1.0, min(60.0, float(seconds)))
        logger.info(f"Simulation engine interval updated to: {self.interval_seconds}s")

    def reset_stats(self):
        self.total_generated = 0
        self.scenario_stats = {k: 0 for k in self.scenario_stats}
        self.recent_events.clear()
        logger.info("Simulation engine metrics reset.")

    def _random_indian_phone(self) -> str:
        prefix = random.choice(["98", "97", "99", "91", "88", "87", "70"])
        rest = "".join([str(random.randint(0, 9)) for _ in range(8)])
        return f"+91{prefix}{rest}"

    def build_event_payload(self, scenario: str) -> tuple[str, dict]:
        """Generate realistic event payload for specified attack or normal scenario."""
        uid = uuid.uuid4().hex[:8]
        entity_id = f"cust_{random.randint(100, 999)}_{uid[:4]}"

        if scenario == "digital_arrest":
            phone = self._random_indian_phone()
            payload = {
                "id": f"call_vishing_{uid}",
                "caller_phone": phone,
                "linked_account_id": entity_id,
                "transcript": random.choice(DIGITAL_ARREST_TRANSCRIPTS),
                "duration_seconds": random.randint(90, 240),
                "stir_shaken_attestation": "C",
                "line_type": "NON_FIXED_VOIP",
                "hour_of_day": random.choice([22, 23, 0, 1]),
                "complaint_history_count": random.randint(2, 6),
            }
            return "call", payload

        elif scenario == "banking_otp":
            phone = self._random_indian_phone()
            payload = {
                "id": f"call_otp_{uid}",
                "caller_phone": phone,
                "linked_account_id": entity_id,
                "transcript": random.choice(BANKING_OTP_TRANSCRIPTS),
                "duration_seconds": random.randint(45, 90),
                "stir_shaken_attestation": "C",
                "line_type": "NON_FIXED_VOIP",
                "hour_of_day": random.choice([20, 21, 22]),
                "complaint_history_count": random.randint(1, 4),
            }
            return "call", payload

        elif scenario == "boiler_room":
            payload = {
                "id": f"call_boiler_{uid}",
                "caller_phone": self._boiler_room_caller,
                "linked_account_id": entity_id,
                "transcript": random.choice(BOILER_ROOM_TRANSCRIPTS),
                "duration_seconds": random.randint(30, 60),
                "stir_shaken_attestation": "C",
                "line_type": "NON_FIXED_VOIP",
                "hour_of_day": 22,
                "call_velocity_1h": random.randint(80, 150),
            }
            return "call", payload

        elif scenario == "voice_clone":
            phone = self._random_indian_phone()
            payload = {
                "id": f"call_deepfake_{uid}",
                "caller_phone": phone,
                "linked_account_id": entity_id,
                "transcript": random.choice(VOICE_CLONE_TRANSCRIPTS),
                "duration_seconds": random.randint(60, 180),
                "stir_shaken_attestation": "B",
                "line_type": "VOIP",
                "hour_of_day": 16,
            }
            return "call", payload

        elif scenario == "blocklist_trigger":
            # Uses known blocklisted number to trigger sub-millisecond Layer 4 short-circuit
            payload = {
                "id": f"call_blocked_{uid}",
                "caller_phone": "+919876543210",
                "linked_account_id": entity_id,
                "transcript": "Hello, customer service calling regarding your account.",
                "duration_seconds": 15,
                "stir_shaken_attestation": "C",
                "line_type": "NON_FIXED_VOIP",
                "hour_of_day": 14,
            }
            return "call", payload

        elif scenario == "cross_channel":
            # Generates high risk transaction following suspect call
            payload = {
                "id": f"txn_heist_{uid}",
                "account_id": entity_id,
                "customer_id": entity_id,
                "amount": round(random.uniform(75000, 250000), 2),
                "currency": "INR",
                "merchant": "Crypto Exchange Global LLC",
                "ip_address": f"103.{random.randint(10, 250)}.{random.randint(1, 250)}.{random.randint(1, 250)}",
                "is_new_payee": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return "transaction", payload

        else:  # benign_call default
            phone = self._random_indian_phone()
            payload = {
                "id": f"call_normal_{uid}",
                "caller_phone": phone,
                "linked_account_id": entity_id,
                "transcript": random.choice(BENIGN_TRANSCRIPTS),
                "duration_seconds": random.randint(30, 120),
                "stir_shaken_attestation": "A",
                "line_type": "MOBILE",
                "hour_of_day": random.randint(9, 18),
                "complaint_history_count": 0,
            }
            return "call", payload

    def process_and_record_event(self, scenario: str) -> Dict[str, Any]:
        """Dispatch event through full 5-Layer orchestrator and save telemetry entry."""
        channel, payload = self.build_event_payload(scenario)
        t_start = time.perf_counter()

        try:
            result = handle_event(channel, payload)
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            case_id = result.get("case_id") or "N/A"
            case_risk = result.get("case_risk") or {}
            risk_score = round(case_risk.get("risk_score") or 0.0, 3)

            # Determine risk level
            if risk_score >= 0.7 or result.get("final_status") == "failed":
                risk_level = "CRITICAL" if risk_score >= 0.9 else "HIGH"
            elif risk_score >= 0.4 or result.get("final_status") == "warning":
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW" if risk_score > 0 else "BENIGN"

            entry = {
                "event_id": payload.get("id"),
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "scenario": scenario,
                "channel": channel,
                "case_id": case_id,
                "entity_id": payload.get("linked_account_id") or payload.get("account_id") or "UNKNOWN",
                "caller_phone": payload.get("caller_phone") or "N/A",
                "transcript": payload.get("transcript") or f"Transaction: {payload.get('merchant', 'N/A')} amount={payload.get('amount')}",
                "risk_score": risk_score,
                "risk_level": risk_level,
                "final_status": result.get("final_status", "success"),
                "duration_ms": duration_ms,
                "violations_count": len(result.get("violations", [])),
                "violations": result.get("violations", []),
            }

            self.recent_events.appendleft(entry)
            self.total_generated += 1
            if scenario in self.scenario_stats:
                self.scenario_stats[scenario] += 1

            logger.info(
                f"[SIMULATION] Scenario '{scenario}' -> Case: {case_id[:8]}.. | Risk: {risk_score} ({risk_level}) | Latency: {duration_ms}ms"
            )
            return entry

        except Exception as e:
            logger.error(f"[SIMULATION ERROR] Failed processing scenario '{scenario}': {e}")
            return {"error": str(e), "scenario": scenario}

    def inject_scenario(self, scenario: str) -> Dict[str, Any]:
        """Manually trigger a specific attack or benign scenario on demand."""
        valid_scenarios = [
            "digital_arrest",
            "banking_otp",
            "boiler_room",
            "voice_clone",
            "blocklist_trigger",
            "benign_call",
            "cross_channel",
        ]
        chosen = scenario if scenario in valid_scenarios else "benign_call"
        return self.process_and_record_event(chosen)

    def _simulation_worker(self):
        """Background daemon generating realistic threat traffic at regular interval."""
        logger.info("Simulation worker thread started.")
        scenarios_pool = [
            ("benign_call", 0.40),
            ("digital_arrest", 0.20),
            ("banking_otp", 0.15),
            ("boiler_room", 0.10),
            ("voice_clone", 0.05),
            ("blocklist_trigger", 0.05),
            ("cross_channel", 0.05),
        ]
        scenarios, weights = zip(*scenarios_pool)

        # Pre-seed with a couple initial diverse events so dashboard is rich immediately
        for init_scen in ["digital_arrest", "banking_otp", "benign_call", "boiler_room"]:
            try:
                self.process_and_record_event(init_scen)
            except Exception as e:
                logger.warning(f"Initial seed warning: {e}")
            time.sleep(0.5)

        while not self._stop_event.is_set():
            if self.is_running:
                chosen = random.choices(scenarios, weights=weights, k=1)[0]
                self.process_and_record_event(chosen)

            # Controllable sleep
            sleep_time = self.interval_seconds
            step = 0.5
            elapsed = 0.0
            while elapsed < sleep_time and not self._stop_event.is_set():
                time.sleep(step)
                elapsed += step

    def start_background_simulation(self):
        """Start the background generation thread if not already running."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._simulation_worker, daemon=True, name="SOCSimulationWorker")
                self._thread.start()
                logger.info("Background SOC simulation daemon launched.")

    def stop_background_simulation(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Background SOC simulation stopped.")


_sim_engine_instance = None


def get_simulation_engine() -> SimulationEngine:
    global _sim_engine_instance
    if _sim_engine_instance is None:
        _sim_engine_instance = SimulationEngine()
    return _sim_engine_instance
