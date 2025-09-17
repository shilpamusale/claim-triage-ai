import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def create_mock_inference_data_v5():
    """
    Generates a diversified mock dataset for inference with realistic,
    correlated denial scenarios. Ground-truth denial fields are dropped.
    """
    num_records = 600
    print(f"Generating {num_records} diversified records for inference...")

    data = []
    
    # Denial categories
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
        "auth": ["Follow-up with payer; no auth found.", "Retro-auth request faxed.", "Escalated to auth team."],
        "necessity": ["Peer-to-peer scheduled.", "Appeal letter submitted.", "Clinical review notes sent."],
        "misc": ["COB updated with primary payer.", "Duplicate claim voided.", "Corrected claim pending resubmission."],
        "eligibility": ["Eligibility confirmed with payer rep.", "Coverage terminated letter received.", "Patient asked for updated ID card."],
        "coding": ["Coding team reviewing CPT/ICD mapping.", "Modifier correction submitted.", "Re-billed with corrected diagnosis code."]
    }

    denial_categories = list(denial_reasons.keys())

    for i in range(1, num_records + 1):
        # Default: clean claim
        is_denied_pattern = False
        denial_reason = None
        followup_note = "Clean claim, auto-adjudicated."
        prior_auth = 'Y'
        payer = f"PAYER_{random.choice(['A', 'B', 'C', 'D'])}"
        total_charge = round(random.uniform(100.0, 800.0), 2)

        # Weighted probabilities of denials
        rand_val = np.random.rand()
        if rand_val < 0.20:  # Auth
            is_denied_pattern = True
            denial_reason = random.choice(denial_reasons["auth"])
            followup_note = random.choice(notes_map["auth"])
            prior_auth = 'N'
            total_charge = round(random.uniform(500.0, 2000.0), 2)
        elif rand_val < 0.40:  # Necessity
            is_denied_pattern = True
            denial_reason = random.choice(denial_reasons["necessity"])
            followup_note = random.choice(notes_map["necessity"])
            payer = random.choice(['PAYER_D_HIGH_DENIAL', 'PAYER_E_STRICT'])
        elif rand_val < 0.55:  # Misc
            is_denied_pattern = True
            denial_reason = random.choice(denial_reasons["misc"])
            followup_note = random.choice(notes_map["misc"])
        elif rand_val < 0.70:  # Eligibility
            is_denied_pattern = True
            denial_reason = random.choice(denial_reasons["eligibility"])
            followup_note = random.choice(notes_map["eligibility"])
        elif rand_val < 0.85:  # Coding
            is_denied_pattern = True
            denial_reason = random.choice(denial_reasons["coding"])
            followup_note = random.choice(notes_map["coding"])
        # else: remains a clean claim

        # Dates
        service_date = datetime.now() - timedelta(days=random.randint(30, 365))
        submission_date = service_date + timedelta(days=random.randint(1, 10))
        patient_dob = datetime.now() - timedelta(days=random.randint(365*20, 365*70))

        record = {
            'claim_id': 1000 + i,
            'submission_date': submission_date.strftime('%Y-%m-%d'),
            'service_date': service_date.strftime('%Y-%m-%d'),
            'denial_date': None,  # always None for inference
            'denied': 1 if is_denied_pattern else 0,  # internal ground truth
            'denial_reason': denial_reason,
            'payer_id': payer,
            'provider_type': random.choice(['Cardiology', 'Oncology', 'Primary Care', 'Orthopedics', 'Radiology']),
            'resubmission': random.choice([0, 1]),
            'followup_notes': followup_note,
            'total_charge_amount': total_charge,
            'patient_dob': patient_dob.strftime('%Y-%m-%d'),
            'prior_authorization': prior_auth,
            'accident_indicator': random.choice(['Y', 'N']),
            'claim_type': random.choice(['Professional', 'Institutional']),
            'billing_provider_specialty': random.choice(['Cardiology', 'Oncology', 'Primary Care', 'Orthopedics', 'Radiology']),
            'facility_code': random.choice([11, 21, 22, 23]),
            'plan_type': random.choice(['private', 'medicare', 'medicaid']),
            'diagnosis_code_primary': random.choice(['Z00.00', 'E11.9', 'M54.5', 'I10', 'J44.9']),
            'procedure_code': random.choice([99213, 99214, 85025, 93000, 99203, 99204, 36415])
        }
        data.append(record)

    df_full = pd.DataFrame(data)

    # --- Drop ground truth columns for inference ---
    cols_to_drop = ['denied', 'denial_date']
    df_inference = df_full.drop(columns=cols_to_drop)

    output_filename = 'data/inference_claims.csv'
    df_inference.to_csv(output_filename, index=False)

    print(f"\nFile '{output_filename}' created with diversified inference data.")
    print(f"Ground truth columns {cols_to_drop} were removed.")

if __name__ == "__main__":
    create_mock_inference_data_v5()
