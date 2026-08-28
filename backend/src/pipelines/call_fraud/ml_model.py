import os
import logging
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier

from backend.src.pipelines.call_fraud.ml_features import CallFeatures

logger = logging.getLogger("call-fraud-ml")


class CallFraudMLModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CallFraudMLModel, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        """Initialize and fit a calibrated Random Forest classifier on call fraud features."""
        logger.info("Initializing Call Fraud Machine Learning Classifier...")
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        
        # Synthetic baseline training set covering call fraud patterns
        np.random.seed(42)
        n_samples = 600

        # Features: [urgency, otp_flag, impersonation, fin_demand, is_spoof, stir_shaken_risk, is_voip_line, fanout_ratio, velocity_norm, cross_acc_norm, duration_norm, complaints, off_hours]
        X = []
        y = []

        for _ in range(n_samples):
            # Legitimate call sample
            if np.random.rand() > 0.5:
                urgency = np.random.uniform(0.0, 0.2)
                otp = 0
                impersonation = np.random.uniform(0.0, 0.1)
                fin_demand = np.random.uniform(0.0, 0.3)
                is_spoof = 0 if np.random.rand() > 0.05 else 1
                stir_shaken = 0.0 if np.random.rand() > 0.1 else 0.3  # Mostly Attestation A/B
                is_voip = 0 if np.random.rand() > 0.1 else 1  # Mobile/Landline
                fanout = np.random.uniform(0.0, 0.2)  # Low fan-out (calling 1 target repeatedly)
                velocity = np.random.uniform(0.0, 0.2)  # Low velocity
                cross_acc = np.random.uniform(0.0, 0.2)  # Single account target
                dur = np.random.uniform(0.2, 1.0)
                complaints = 0 if np.random.rand() > 0.9 else 0
                off_hours = 0 if np.random.rand() > 0.2 else 1
                X.append([urgency, otp, impersonation, fin_demand, is_spoof, stir_shaken, is_voip, fanout, velocity, cross_acc, dur, complaints, off_hours])
                y.append(0)
            # Fraudulent vishing call sample
            else:
                urgency = np.random.uniform(0.4, 1.0)
                otp = 1 if np.random.rand() > 0.3 else 0
                impersonation = np.random.uniform(0.5, 1.0)
                fin_demand = np.random.uniform(0.4, 1.0)
                is_spoof = 1 if np.random.rand() > 0.4 else 0
                stir_shaken = np.random.choice([0.8, 1.0])  # Gateway C or Failed PASSporT
                is_voip = 1 if np.random.rand() > 0.3 else 0  # High VOIP ratio
                fanout = np.random.uniform(0.6, 1.0)  # High fan-out ratio (boiler room burst)
                velocity = np.random.uniform(0.5, 1.0)  # High velocity
                cross_acc = np.random.uniform(0.4, 1.0)  # Multi-account targeted
                dur = np.random.uniform(0.05, 0.5)
                complaints = np.random.randint(1, 8)
                off_hours = 1 if np.random.rand() > 0.4 else 0
                X.append([urgency, otp, impersonation, fin_demand, is_spoof, stir_shaken, is_voip, fanout, velocity, cross_acc, dur, complaints, off_hours])
                y.append(1)



        X = np.array(X)
        y = np.array(y)
        self.clf.fit(X, y)
        self.feature_names = CallFeatures.feature_names()
        
        # Retrain on real historical cases from SQL database if available
        self.retrain_from_database_history()
        logger.info("Call Fraud ML Classifier ready.")

    def retrain_from_database_history(self) -> int:
        """
        Retrains or updates the model using real historical case outcomes and HITL analyst dispositions
        persisted in the CaseStore database. Returns count of real database cases ingested.
        """
        try:
            from backend.src.case.models import Case, CaseEvent, CaseStatus
            from backend.src.case.db import get_session
            session = get_session()
            real_cases = session.query(Case).all()
            
            real_X = []
            real_y = []
            
            for c in real_cases:
                # Get case events to extract feature vector
                events = session.query(CaseEvent).filter(CaseEvent.case_id == c.case_id).all()
                if not events:
                    continue
                
                latest_res = events[-1].pipeline_result or {}
                feat_dict = latest_res.get("features", {})
                if not feat_dict:
                    continue
                
                # Determine ground truth label from status
                if c.status in [CaseStatus.ESCALATED.value, "BLOCKED", "CONFIRMED_FRAUD"]:
                    label = 1
                elif c.status in [CaseStatus.CLOSED.value, "FALSE_POSITIVE", "CLEARED"]:
                    label = 0
                elif c.risk_score >= 0.6:
                    label = 1
                else:
                    label = 0
                    
                vec = [
                    feat_dict.get("urgency_intent_score", 0.0),
                    feat_dict.get("otp_request_flag", 0),
                    feat_dict.get("impersonation_score", 0.0),
                    feat_dict.get("financial_demand_score", 0.0),
                    feat_dict.get("caller_spoof_flag", 0),
                    feat_dict.get("stir_shaken_risk", 0.0),
                    feat_dict.get("is_voip_line", 0),
                    feat_dict.get("fanout_ratio_1h", 0.0),
                    feat_dict.get("call_velocity_1h_norm", 0.0),
                    feat_dict.get("cross_account_target_norm", 0.0),
                    feat_dict.get("call_duration_norm", 0.0),
                    feat_dict.get("prior_complaints_norm", 0.0),
                    feat_dict.get("off_hours_flag", 0),
                ]
                if len(vec) == 13:
                    real_X.append(vec)
                    real_y.append(label)

            session.close()

            if real_X and len(real_X) >= 5:
                logger.info(f"Retraining Call Fraud ML Classifier on {len(real_X)} real database cases...")
                self.clf.fit(np.array(real_X), np.array(real_y))
                return len(real_X)
        except Exception as e:
            logger.warning(f"ML Model: Could not retrain on DB history: {e}")
        return 0

    def predict(self, features: CallFeatures) -> Dict[str, Any]:
        """
        Run ML inference on extracted call features.
        Returns probability, risk level, top feature drivers, and recommended automated action.
        """
        vector = np.array(features.to_feature_vector()).reshape(1, -1)
        prob = float(self.clf.predict_proba(vector)[0][1])

        # Risk categorization
        if prob >= 0.80:
            risk_level = "CRITICAL"
            action = "BLOCK_CALL_AND_FREEZE_ACCOUNT"
        elif prob >= 0.55:
            risk_level = "HIGH"
            action = "FLAG_SUSPICIOUS_AND_ALERT_I4C"
        elif prob >= 0.30:
            risk_level = "MEDIUM"
            action = "WARN_USER_IN_REALTIME"
        else:
            risk_level = "LOW"
            action = "ALLOW_CALL"

        # Calculate feature contributions for explainable ML
        feature_weights = self.clf.feature_importances_
        feature_values = features.to_feature_vector()
        
        contributions = []
        for name, val, weight in zip(self.feature_names, feature_values, feature_weights):
            if val > 0:
                impact = round(float(val * weight * 100), 2)
                contributions.append((name, impact, val))
        
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        top_drivers = [
            f"{c[0].replace('_', ' ').title()} (val={c[2]}, impact={c[1]}%)"
            for c in contributions[:3]
        ]

        return {
            "fraud_probability": round(prob, 4),
            "fraud_percentage": round(prob * 100, 1),
            "risk_level": risk_level,
            "recommended_action": action,
            "top_risk_drivers": top_drivers,
            "model_type": "RandomForestClassifier (Pre-trained Baseline + Online HITL Retrained)",
        }


# Global singleton instance
_ml_model = None

def get_call_ml_model() -> CallFraudMLModel:
    global _ml_model
    if _ml_model is None:
        _ml_model = CallFraudMLModel()
    return _ml_model
