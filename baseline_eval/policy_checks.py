"""
Policy validation checks for baseline evaluation.
Detects violations, compliance, and ambiguity.
"""

# ----------------------------
# Constants
# ----------------------------
REGIONS = {
    "US": {"return_window": 60},
    "CA": {"return_window": 60},
    "EU": {"return_window": 30},
    "UK": {"return_window": 30},
    "AU": {"return_window": 45},
}

HIGH_VALUE_FRAUD_THRESHOLD = 500


# ----------------------------
# Helper
# ----------------------------
def missing(ctx, *fields):
    return any(field not in ctx for field in fields)


# ----------------------------
# Policy Checks
# ----------------------------
def check_return_policy(input_data, output):
    issue_type = input_data.get("issue_type")
    if issue_type not in ["return_exchange", "size_fit"]:
        return True, None, None

    ctx = input_data.get("issue_context", {})
    decision = output.get("decision")
    region = input_data.get("region", "US")

    if missing(ctx, "days_since_delivery", "condition"):
        return True, "ambiguous", "Missing return context fields"

    days = ctx["days_since_delivery"]
    condition = ctx["condition"]
    rw = REGIONS.get(region, REGIONS["US"])["return_window"]

    if condition == "defective" and decision != "Eligible":
        return False, "violation", "Defective items must be eligible"

    if days > rw and condition != "defective" and decision == "Eligible":
        return False, "violation", "Return approved outside window"

    if ctx.get("final_sale") and condition != "defective" and decision == "Eligible":
        return False, "violation", "Final sale item approved incorrectly"

    if days <= rw and condition not in ["unworn", "tried_indoors", "defective"]:
        if decision == "Eligible":
            return False, "violation", "Invalid condition approved"

    return True, None, None


def check_shipping_delay_policy(input_data, output):
    if input_data.get("issue_type") != "shipping_delay":
        return True, None, None

    ctx = input_data.get("issue_context", {})
    decision = output.get("decision")

    if missing(ctx, "days_delayed"):
        return True, "ambiguous", "Missing delay information"

    days = ctx["days_delayed"]

    if ctx.get("our_error") and decision != "Eligible":
        return False, "violation", "Our error must be eligible"

    if days >= 7 and not ctx.get("weather_related"):
        if decision not in ["Eligible", "In Progress"]:
            return False, "violation", "Delay should be compensated"

    return True, None, None


def check_fraud_policy(input_data, output):
    if input_data.get("issue_type") != "fraud_chargeback":
        return True, None, None

    ctx = input_data.get("issue_context", {})
    decision = output.get("decision")

    if missing(ctx, "identity_verified", "transaction_amount"):
        return True, "ambiguous", "Missing fraud verification details"

    if not ctx["identity_verified"] and decision == "Eligible":
        return False, "violation", "Fraud approved without verification"

    if ctx["transaction_amount"] > HIGH_VALUE_FRAUD_THRESHOLD and decision != "Escalate":
        return False, "violation", "High-value fraud must escalate"

    return True, None, None


def check_warranty_policy(input_data, output):
    if input_data.get("issue_type") != "warranty_claim":
        return True, None, None

    ctx = input_data.get("issue_context", {})
    decision = output.get("decision")

    if missing(ctx, "within_warranty", "defect_type"):
        return True, "ambiguous", "Missing warranty details"

    if not ctx["within_warranty"] and decision == "Eligible":
        return False, "violation", "Warranty approved outside coverage"

    if ctx["defect_type"] == "normal_wear" and decision == "Eligible":
        return False, "violation", "Normal wear incorrectly approved"

    return True, None, None


def check_payment_policy(input_data, output):
    if input_data.get("issue_type") != "payment_issue":
        return True, None, None

    ctx = input_data.get("issue_context", {})
    decision = output.get("decision")

    if missing(ctx, "status"):
        return True, "ambiguous", "Missing payment status"

    if ctx["status"] == "duplicate_charge" and decision != "Eligible":
        return False, "violation", "Duplicate charge must be eligible"

    if ctx["status"] == "pending":
        days_pending = ctx.get("days_pending") or 0
        if days_pending and days_pending > 5:
            if decision != "Escalate":
                return False, "violation", "Long pending charge must escalate"

    return True, None, None


def check_order_modification_policy(input_data, output):
    if input_data.get("issue_type") != "order_modification":
        return True, None, None

    ctx = input_data.get("issue_context", {})
    decision = output.get("decision")

    if missing(ctx, "fulfillment_stage"):
        return True, "ambiguous", "Missing fulfillment stage"

    if ctx["fulfillment_stage"] in ["shipped", "delivered"] and decision == "Eligible":
        return False, "violation", "Modification approved after shipment"

    return True, None, None


def check_promotion_pricing_policy(input_data, output):
    if input_data.get("issue_type") != "promotion_pricing":
        return True, None, None

    ctx = input_data.get("issue_context", {})
    decision = output.get("decision")

    if ctx.get("stack_attempt") and decision == "Eligible":
        return False, "violation", "Discount stacking not allowed"

    if ctx.get("promo_start_offset_days") is not None:
        if ctx["promo_start_offset_days"] < 0 and decision == "Eligible":
            return False, "violation", "Retroactive promotion approved"

    return True, None, None


def check_all_policies(input_data, output):
    checks = [
        check_return_policy,
        check_shipping_delay_policy,
        check_fraud_policy,
        check_warranty_policy,
        check_payment_policy,
        check_order_modification_policy,
        check_promotion_pricing_policy,
    ]

    for check in checks:
        compliant, violation_type, details = check(input_data, output)
        if not compliant or violation_type == "ambiguous":
            return compliant, violation_type, details

    return True, None, None
