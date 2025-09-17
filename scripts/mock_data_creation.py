import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def create_mock_training_data_v5():
    """
    Generates a highly diversified mock dataset with correlated,
    realistic denial scenarios.
    """
    num_records = 6000
    print(f"Generating {num_records} diversified claim records...")

    data = []
    
    # Denial reason categories
    denial_reasons = {
        "auth": [
            "Prior Authorization Required but Not Obtained",
            "Authorization Expired on Service Date",
            "Pre-certification for procedure is missing",
            "Referral not on file",
            "Auth number invalid"
        ],
        "necessity": [
            "Service Not Deemed Medically Necessary per Policy XYZ-123",
            "Procedure Inconsistent with Diagnosis Code",
            "Exceeds Plan Limitation for this service type",
            "Level of care not appropriate",
            "Experimental or Investigational Treatment"
        ],
        "misc": [
            "Service is considered Bundled, Not Separately Billable",
            "Patient ID Mismatch with Payer Records",
            "Duplicate Claim Submission Detected",
            "Coordination of Benefits issue",
            "Timely Filing Limit Exceeded"
        ],
        "eligibility": [
            "Member Coverage Terminated before Service Date",
            "Service Not Covered under Current Plan",
            "Out-of-Network Provider for Plan",
            "Coverage Lapsed due to Non-Payment of Premium",
            "Service Requires Plan Upgrade"
        ],
        "coding": [
            "Invalid Procedure Code Submitted",
            "Diagnosis Code Not Supported",
            "Procedure-Diagnosis Mismatch",
            "Modifier Missing or Invalid",
            "Unbundled Service Detected"
        ]
    }
    
    notes_map = {
        "auth": ["Auth team contacted; no auth found.", "Retro-auth requested.", "Escalated to payer auth desk."],
        "necessity": ["Appeal letter drafted.", "Peer-to-peer review scheduled.", "Clinical notes submitted for review."],
        "misc": ["Billing team correcting COB info.", "Duplicate claim voided.", "Corrected claim pending resubmission."],
        "eligibility": ["Eligibility verified with payer.", "Patient asked to provide updated ID card.", "Coverage terminated notice received."],
        "coding": ["Coding team reviewing ICD/CPT mapping.", "Modifier correction submitted.", "Re-billed with corrected diagnosis code."]
    }

    denial_categories = list(denial_reasons.keys())

    for i in range(1, num_records + 1):
        # Default: paid claim
        is_denied = False; denial_reason = None; followup_note = None
        prior_auth = 'Y'; payer = f"PAYER_{random.choice(['A', 'B', 'C', 'D'])}"
        total_charge = round(random.uniform(100.0, 800.0), 2)

        # Weighted scenario probabilities
        rand_val = np.random.rand()
        if rand_val < 0.25:  # Auth issues
            is_denied = True
            denial_reason = random.choice(denial_reasons["auth"])
            followup_note = random.choice(notes_map["auth"])
            prior_auth = 'N'
            total_charge = round(random.uniform(500.0, 2000.0), 2)
        elif rand_val < 0.45:  # Necessity
            is_denied = True
            denial_reason = random.choice(denial_reasons["necessity"])
            followup_note = random.choice(notes_map["necessity"])
            payer = random.choice(['PAYER_D_HIGH_DENIAL', 'PAYER_E_STRICT'])
        elif rand_val < 0.55:  # Misc
            is_denied = True
            denial_reason = random.choice(denial_reasons["misc"])
            followup_note = random.choice(notes_map["misc"])
        elif rand_val < 0.70:  # Eligibility
            is_denied = True
            denial_reason = random.choice(denial_reasons["eligibility"])
            followup_note = random.choice(notes_map["eligibility"])
        elif rand_val < 0.85:  # Coding
            is_denied = True
            denial_reason = random.choice(denial_reasons["coding"])
            followup_note = random.choice(notes_map["coding"])

        # Dates
        service_date = datetime.now() - timedelta(days=random.randint(30, 365))
        submission_date = service_date + timedelta(days=random.randint(1, 10))
        patient_dob = datetime.now() - timedelta(days=random.randint(365*20, 365*70))
        denial_date = (service_date + timedelta(days=random.randint(2, 40))).strftime('%Y-%m-%d') if is_denied else None

        record = {
            'claim_id': i,
            'denied': 1 if is_denied else 0,
            'denial_reason': denial_reason,
            'followup_notes': followup_note,
            'payer_id': payer,
            'prior_authorization': prior_auth,
            'total_charge_amount': total_charge,
            'submission_date': submission_date.strftime('%Y-%m-%d'),
            'service_date': service_date.strftime('%Y-%m-%d'),
            'denial_date': denial_date,
            'provider_type': random.choice(['Cardiology', 'Oncology', 'Primary Care', 'Orthopedics', 'Radiology']),
            'resubmission': random.choice([0, 1]),
            'patient_dob': patient_dob.strftime('%Y-%m-%d'),
            'accident_indicator': random.choice(['Y', 'N']),
            'claim_type': random.choice(['Professional', 'Institutional']),
            'billing_provider_specialty': random.choice(['Cardiology', 'Oncology', 'Primary Care', 'Orthopedics', 'Radiology']),
            'facility_code': random.choice([11, 21, 22, 23]),
            'plan_type': random.choice(['private', 'medicare', 'medicaid']),
            'diagnosis_code_primary': random.choice(['Z00.00', 'E11.9', 'M54.5', 'I10', 'J44.9']),
            'procedure_code': random.choice([99213, 99214, 85025, 93000, 99203, 99204, 36415])
        }
        data.append(record)

    df_training = pd.DataFrame(data)
    df_training.to_csv('data/raw_claims_v2.csv', index=False)
    print(f"\nFile 'raw_claims_v3.csv' created with diversified denial scenarios.")

if __name__ == "__main__":
    create_mock_training_data_v5()
