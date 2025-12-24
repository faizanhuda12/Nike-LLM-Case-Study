"""
Generate SFT Training Dataset from Policy Documents
Generates 10,000 training examples based on actual policy documents
Includes edge cases, boundary conditions, and complex scenarios
"""

import json
import random
import string
import datetime
import os
from pathlib import Path

random.seed(123)

# Configuration
N = 10000  # Total training samples
POLICIES_DIR = "policies"

# Regions and their return windows
regions = {
    "US": {"return_window": 60, "tax_type": "sales_tax"},
    "CA": {"return_window": 60, "tax_type": "gst_hst"},
    "EU": {"return_window": 30, "tax_type": "vat"},
    "UK": {"return_window": 30, "tax_type": "vat"},
    "AU": {"return_window": 45, "tax_type": "gst"}
}

channels = ["chat", "email", "agent_assist", "ivr"]
tiers = ["Guest", "Member", "VIP", "Employee"]
product_categories = ["Footwear", "Apparel", "Accessories", "Equipment", "Digital"]

# Issue types based on policy documents
issue_types = [
    "return_exchange",
    "promotion_pricing",
    "shipping_delay",
    "address_issue",
    "order_modification",
    "loyalty_points",
    "damaged_item",
    "missing_item",
    "size_fit",
    "payment_issue",
    "fraud_chargeback",
    "warranty_claim",
    "preorder_delay",
    "store_pickup",
    "price_match",
    "subscription_membership",
    "digital_product_issue",
    "tax_duty_question",
    "compliance_restriction"
]

# Load policy documents
def load_policy(policy_name):
    """Load a policy document"""
    policy_path = Path(POLICIES_DIR) / f"{policy_name}.txt"
    if policy_path.exists():
        with open(policy_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Load all policies
policies = {
    "returns": load_policy("returns_and_exchanges_policy"),
    "shipping": load_policy("shipping_and_delivery_policy"),
    "promotions": load_policy("promotions_and_pricing_policy"),
    "payment": load_policy("payment_processing_policy"),
    "fraud": load_policy("fraud_and_chargeback_policy"),
    "warranty": load_policy("warranty_policy"),
    "tax": load_policy("tax_and_duty_policy"),
    "order_mod": load_policy("order_modification_policy"),
    "loyalty": load_policy("loyalty_points_policy"),
    "digital": load_policy("digital_products_policy"),
    "compliance": load_policy("compliance_and_restrictions_policy"),
    "tone": load_policy("customer_service_tone_and_communication_policy")
}

def oid():
    """Generate order ID"""
    return "ORD-" + "".join(random.choices(string.digits, k=7))

def sku():
    """Generate SKU"""
    return "SKU-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

def pick(x):
    """Pick random element"""
    return random.choice(x)

def get_relevant_policy(issue_type):
    """Get relevant policy text for issue type"""
    policy_map = {
        "return_exchange": policies["returns"],
        "promotion_pricing": policies["promotions"],
        "shipping_delay": policies["shipping"],
        "address_issue": policies["shipping"],
        "order_modification": policies["order_mod"],
        "loyalty_points": policies["loyalty"],
        "damaged_item": policies["returns"] + "\n\n" + policies["shipping"],
        "missing_item": policies["shipping"],
        "size_fit": policies["returns"],
        "payment_issue": policies["payment"],
        "fraud_chargeback": policies["fraud"],
        "warranty_claim": policies["warranty"],
        "preorder_delay": policies["shipping"],
        "store_pickup": policies["shipping"],
        "price_match": policies["promotions"],
        "subscription_membership": policies["digital"],
        "digital_product_issue": policies["digital"],
        "tax_duty_question": policies["tax"],
        "compliance_restriction": policies["compliance"]
    }
    return policy_map.get(issue_type, "")

def generate_context(issue_type, region):
    """Generate realistic context for issue type"""
    rw = regions[region]["return_window"]
    
    if issue_type == "return_exchange":
        # Edge cases: exactly at window, just outside, defective, etc.
        days_options = [
            rw - 1,  # Just within window
            rw,      # Exactly at window
            rw + 1,  # Just outside
            rw + 5,  # Outside window
            rw + 30, # Well outside
            0,       # Just delivered
        ]
        conditions = ["unworn", "tried_indoors", "worn_outdoors", "defective", "damaged_in_transit", "tags_removed"]
        return {
            "days_since_delivery": pick(days_options),
            "condition": pick(conditions),
            "tags_attached": pick([True, False]),
            "original_packaging": pick([True, False]),
            "final_sale": pick([True, False]) if random.random() < 0.1 else False,
            "holiday_purchase": pick([True, False]) if random.random() < 0.15 else False
        }
    
    elif issue_type == "size_fit":
        # Similar to return_exchange but focused on size/fit issues
        days_options = [
            rw - 1, rw, rw + 1, rw + 5, rw + 30, 0
        ]
        conditions = ["unworn", "tried_indoors", "worn_outdoors"]
        return {
            "days_since_delivery": pick(days_options),
            "condition": pick(conditions),
            "tags_attached": pick([True, False]),
            "exchange_requested": pick([True, False]),
            "size_available": pick([True, False]),
            "final_sale": pick([True, False]) if random.random() < 0.1 else False
        }
    
    elif issue_type == "damaged_item":
        return {
            "days_since_delivery": pick([0, 1, 2, 3, 5, 10, rw - 1, rw, rw + 1]),
            "damage_type": pick(["damaged_in_transit", "defective", "customer_damage"]),
            "photographs_provided": pick([True, False]),
            "packaging_damaged": pick([True, False]),
            "reported_within_48h": pick([True, False])
        }
    
    elif issue_type == "missing_item":
        return {
            "days_since_delivery": pick([0, 1, 2, 3, 5, 7, 10]),
            "package_marked_delivered": pick([True, False]),
            "partial_order": pick([True, False]),
            "carrier_confirmation": pick([True, False]),
            "address_verified": pick([True, False])
        }
    
    elif issue_type == "promotion_pricing":
        return {
            "promo_start_offset_days": pick([-3, -1, 0, 1, 3, 7]),
            "stack_attempt": pick([True, False]),
            "clearance": pick([True, False]),
            "price_protection_window": pick([True, False]) if random.random() < 0.3 else False,
            "student_military_discount": pick([True, False]) if random.random() < 0.2 else False
        }
    
    elif issue_type == "shipping_delay":
        return {
            "days_delayed": pick([1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 20]),
            "carrier": pick(["UPS", "FedEx", "DHL", "USPS"]),
            "weather_related": pick([True, False]),
            "our_error": pick([True, False]) if random.random() < 0.2 else False,
            "carrier_error": pick([True, False]) if random.random() < 0.15 else False,
            "estimated_delivery_date": (datetime.datetime.now() - datetime.timedelta(days=random.randint(5, 20))).isoformat()
        }
    
    elif issue_type == "address_issue":
        return {
            "address_mismatch": pick([True, False]),
            "carrier_exception": pick([True, False]),
            "customer_error": pick([True, False]),
            "system_error": pick([True, False]) if random.random() < 0.2 else False,
            "after_shipment": pick([True, False])
        }
    
    elif issue_type == "order_modification":
        stages = ["placed", "processing", "picked", "shipped", "delivered"]
        return {
            "fulfillment_stage": pick(stages),
            "change_type": pick(["cancel", "size", "color", "address", "add_item", "remove_item"]),
            "hours_since_order": random.randint(0, 48),
            "preorder": pick([True, False]) if random.random() < 0.1 else False
        }
    
    elif issue_type == "loyalty_points":
        return {
            "points_posted": pick([True, False]),
            "days_since_ship": pick([1, 2, 3, 4, 5, 7, 10, 14]),
            "points_expected": random.randint(100, 5000),
            "tier": pick(tiers),
            "return_involved": pick([True, False]) if random.random() < 0.2 else False
        }
    
    elif issue_type == "payment_issue":
    return {
            "payment_method": pick(["credit_card", "debit_card", "paypal", "apple_pay", "gift_card"]),
            "status": pick(["pending", "declined", "duplicate_charge", "authorization_hold", "refund_not_received"]),
            "days_pending": pick([1, 2, 3, 4, 5, 6, 7, 10]) if random.random() < 0.5 else None,
            "refund_days_ago": pick([5, 7, 10, 12, 15]) if random.random() < 0.3 else None
        }
    
    elif issue_type == "fraud_chargeback":
        return {
            "claim_type": pick(["unauthorized_transaction", "chargeback_received", "account_compromise"]),
            "transaction_amount": random.choice([50, 100, 200, 300, 500, 750, 1000, 1500]),
            "identity_verified": pick([True, False]),
            "multiple_transactions": pick([True, False]) if random.random() < 0.3 else False,
            "days_since_transaction": random.randint(1, 30)
        }
    
    elif issue_type == "warranty_claim":
        purchase_days_ago = random.choice([30, 60, 90, 180, 270, 365, 400, 450, 500])
        return {
            "purchase_days_ago": purchase_days_ago,
            "within_warranty": purchase_days_ago <= 365,
            "defect_type": pick(["material_defect", "construction_defect", "sizing_defect", "component_defect", "normal_wear"]),
            "photographs_provided": pick([True, False]),
            "proof_of_purchase": pick([True, False]),
            "item_value": random.choice([50, 100, 200, 300, 500, 750])
        }
    
    elif issue_type == "preorder_delay":
        return {
            "delay_days": pick([3, 7, 14, 21, 30, 45, 60]),
            "supplier_issue": pick([True, False]),
            "estimated_date_passed": pick([True, False]),
            "customer_notified": pick([True, False])
        }
    
    elif issue_type == "store_pickup":
        return {
            "pickup_window_expired": pick([True, False]),
            "days_since_ready": pick([1, 3, 5, 7, 8, 10]),
            "store_stock_issue": pick([True, False]),
            "item_available": pick([True, False])
        }
    
    elif issue_type == "price_match":
        return {
            "competitor_verified": pick([True, False]),
            "within_window": pick([True, False]),
            "days_since_purchase": pick([1, 3, 5, 7, 8, 10]),
            "price_difference": random.choice([5, 10, 15, 20, 25, 50, 100]),
            "authorized_retailer": pick([True, False])
        }
    
    elif issue_type == "subscription_membership":
        return {
            "auto_renew": pick([True, False]),
            "billing_cycle_days": pick([1, 15, 30]),
            "cancellation_requested": pick([True, False]),
            "unused_portion_refund": pick([True, False]) if random.random() < 0.3 else False
        }
    
    elif issue_type == "digital_product_issue":
        return {
            "download_attempts": pick([1, 2, 3, 4, 5]),
            "device_type": pick(["ios", "android", "pc", "mac"]),
            "delivery_received": pick([True, False]),
            "hours_since_purchase": pick([0.5, 1, 2, 6, 12, 24, 48]),
            "product_accessed": pick([True, False]),
            "compatibility_issue": pick([True, False])
        }
    
    elif issue_type == "tax_duty_question":
        return {
            "international": pick([True, False]),
            "duty_charged": pick([True, False]),
            "tax_amount": random.choice([10, 20, 30, 50, 75, 100]),
            "tax_exempt_eligible": pick([True, False]) if random.random() < 0.2 else False,
            "duty_dispute": pick([True, False]) if random.random() < 0.15 else False
        }
    
    elif issue_type == "compliance_restriction":
        return {
            "restricted_region": pick([True, False]),
            "age_verified": pick([True, False]),
            "export_control": pick([True, False]) if random.random() < 0.1 else False,
            "product_restriction": pick([True, False]),
            "compliance_question": pick([True, False])
        }
    
    return {}

def make_decision(issue_type, region, ctx, tier):
    """Make decision based on policy rules"""
    rw = regions[region]["return_window"]
    
    if issue_type == "return_exchange":
        # Defective items always eligible
        if ctx.get("condition") == "defective":
            return "Eligible", "Manufacturing defects are always eligible for return regardless of time or condition."
        
        # Holiday returns extension
        if ctx.get("holiday_purchase") and ctx.get("days_since_delivery", 0) <= 90:
            if ctx.get("condition") in ["unworn", "tried_indoors"]:
                return "Eligible", "Holiday purchase eligible for extended return window until January 31."
        
        # Within return window
        if ctx.get("days_since_delivery", 0) <= rw:
            if ctx.get("condition") in ["unworn", "tried_indoors"] and not ctx.get("final_sale"):
                return "Eligible", f"Item is within the {rw}-day return window and meets condition requirements."
            elif ctx.get("final_sale"):
                return "Not Eligible", "Final sale items are not eligible for return unless defective."
            else:
                return "Not Eligible", "Item condition does not meet return requirements (must be unworn or tried indoors only)."
        
        # Outside return window
        if ctx.get("days_since_delivery", 0) > rw:
            return "Not Eligible", f"Return window of {rw} days has expired. Defective items are always eligible regardless of time."
    
    elif issue_type == "promotion_pricing":
        # Retroactive promotions
        if ctx.get("promo_start_offset_days", 0) < 0:
            # Exception: within 24 hours before promo start
            if ctx.get("promo_start_offset_days") == -1 and random.random() < 0.1:  # 10% exception rate
                return "Escalate", "Order placed within 24 hours before promotion may qualify for exception with supervisor approval."
            return "Not Eligible", "Promotions cannot be applied retroactively to orders placed before the promotion period."
        
        # Price protection window
        if ctx.get("price_protection_window") and ctx.get("promo_start_offset_days", 0) <= 7:
            return "Eligible", "Price protection applies within 7 days of purchase for price drops on same item."
        
        # Code stacking
        if ctx.get("stack_attempt"):
            return "Not Eligible", "Promotional codes cannot be stacked unless explicitly stated in promotion terms."
        
        # Clearance exclusions
        if ctx.get("clearance"):
            return "Escalate", "Clearance item promotion eligibility varies and requires review."
        
        return "Eligible", "Promotion can be applied to this order."
    
    elif issue_type == "shipping_delay":
        delay_days = ctx.get("days_delayed", 0)
        
        # Our error
        if ctx.get("our_error"):
            return "Eligible", "Delay caused by our error. Expedited shipment and shipping fee refund provided."
        
        # Weather-related
        if ctx.get("weather_related") and delay_days < 14:
            return "In Progress", "Weather-related delays are outside our control. Delivery will proceed once conditions permit."
        elif ctx.get("weather_related") and delay_days >= 14:
            return "Eligible", "Weather delay exceeds 14 days. Customer may request cancellation and full refund."
        
        # Delay thresholds
        if delay_days >= 7:
            return "Eligible", "Delay exceeds 7 business days. Shipping fee refund and compensation provided."
        elif delay_days >= 4:
            return "Eligible", "Moderate delay (4-6 days). Shipping fee refund or discount code considered."
        else:
            return "In Progress", "Delay is within normal carrier variance. Monitoring delivery status."
    
    elif issue_type == "address_issue":
        if ctx.get("system_error"):
            return "Eligible", "Address error due to our system. We cover all reshipment costs and expedited shipping."
        elif ctx.get("customer_error"):
            return "Not Eligible", "Customer-provided incorrect address. Customer responsible for additional shipping costs."
        elif ctx.get("after_shipment"):
            return "Escalate", "Address change after shipment requires carrier redirect attempt, not guaranteed."
        else:
            return "Eligible", "Address can be updated before shipment at no cost."
    
    elif issue_type == "order_modification":
        stage = ctx.get("fulfillment_stage", "placed")
        change_type = ctx.get("change_type", "cancel")
        
        if stage == "placed":
            return "Eligible", "Order not yet processed. Modification can be made immediately."
        elif stage == "processing" or stage == "picked":
            if change_type == "cancel":
                return "Escalate", "Order in fulfillment. Cancellation may be possible but requires immediate action."
            else:
                return "Escalate", "Limited modification available after fulfillment begins."
        elif stage == "shipped":
            return "Not Eligible", "Order has shipped. Use return/exchange process after delivery."
        else:
            return "Not Eligible", "Order delivered. Use return/exchange process."
    
    elif issue_type == "loyalty_points":
        days = ctx.get("days_since_ship", 0)
        
        if not ctx.get("points_posted") and days <= 4:
            return "In Progress", "Points typically post within 1-4 business days after shipment. Still within normal processing window."
        elif not ctx.get("points_posted") and days > 7:
            return "Eligible", "Points should have posted. Manual points adjustment will be processed."
        elif ctx.get("return_involved"):
            return "Eligible", "Points adjustment required due to return. Points will be deducted for returned items."
        else:
            return "Eligible", "Points inquiry resolved. Points balance verified."
    
    elif issue_type == "payment_issue":
        status = ctx.get("status", "")
        days_pending = ctx.get("days_pending") or 0
        
        if status == "pending":
            if days_pending <= 5:
                return "In Progress", "Authorization hold may take up to 5 business days to release. This is normal for cancelled orders."
            else:
                return "Escalate", "Pending charge exceeds 5 business days. Escalating to payment specialist."
        elif status == "duplicate_charge":
            return "Eligible", "Duplicate charge confirmed. Refund will be processed within 1 business day."
        elif status == "refund_not_received":
            refund_days = ctx.get("refund_days_ago") or 0
            if refund_days <= 10:
                return "In Progress", f"Refund processing takes 5-10 business days. Still within normal window ({refund_days} days ago)."
            else:
                return "Escalate", "Refund not received after 10+ business days. Escalating to payment specialist."
        elif status == "declined":
            return "Escalate", "Payment decline requires investigation. Customer should contact bank/card issuer."
        else:
            return "Escalate", "Payment issue requires specialist review."
    
    elif issue_type == "fraud_chargeback":
        amount = ctx.get("transaction_amount", 0)
        verified = ctx.get("identity_verified", False)
        
        if not verified:
            return "Escalate", "Identity verification required before processing fraud claim. Multiple verification methods needed."
        
        if amount > 500:
            return "Escalate", "High-value fraud claim requires fraud specialist review."
        
        if ctx.get("claim_type") == "unauthorized_transaction" and verified:
            return "Eligible", "Verified unauthorized transaction. Full refund processed immediately."
        elif ctx.get("claim_type") == "chargeback_received":
            return "Escalate", "Chargeback received. Reviewing with evidence. May contest if order was legitimate."
        else:
            return "Escalate", "Fraud claim requires specialist investigation."
    
    elif issue_type == "warranty_claim":
        within_warranty = ctx.get("within_warranty", False)
        defect_type = ctx.get("defect_type", "")
        has_proof = ctx.get("proof_of_purchase", False)
        
        if not within_warranty:
            return "Not Eligible", "Warranty period is 1 year from purchase. Claim is outside warranty period."
        
        if not has_proof:
            return "Escalate", "Proof of purchase required for warranty claim. May verify through account if possible."
        
        if defect_type == "normal_wear":
            return "Not Eligible", "Normal wear and tear is not covered under warranty. Warranty covers manufacturing defects only."
        elif defect_type in ["material_defect", "construction_defect", "sizing_defect", "component_defect"]:
            item_value = ctx.get("item_value", 0)
            if item_value > 500:
                return "Escalate", "High-value warranty claim requires supervisor review."
            return "Eligible", "Manufacturing defect confirmed. Customer may choose replacement, repair, or refund."
        else:
            return "Escalate", "Warranty claim requires evaluation to determine if defect is manufacturing-related."
    
    elif issue_type == "preorder_delay":
        delay_days = ctx.get("delay_days", 0)
        
        if delay_days <= 7:
            return "In Progress", "Preorder delay within normal variance. Customer will be notified of updated delivery date."
        elif delay_days <= 14:
            return "Eligible", "Preorder delay 8-14 days. 10% discount code provided for inconvenience."
        elif delay_days <= 30:
            return "Eligible", "Preorder delay 15-30 days. 15% discount code or cancellation with full refund available."
        else:
            return "Eligible", "Preorder delay exceeds 30 days. 20% discount code or immediate cancellation with full refund available."
    
    elif issue_type == "store_pickup":
        if ctx.get("pickup_window_expired"):
            days = ctx.get("days_since_ready", 0)
            if days <= 10:
                return "Eligible", "Pickup window expired. One-time extension available or cancellation with full refund."
            else:
                return "Eligible", "Pickup window expired. Order may be cancelled and refunded."
        elif ctx.get("store_stock_issue"):
            return "Eligible", "Item not available at selected store. Options: wait for restock, transfer to another store, or full refund."
        else:
            return "Eligible", "Order ready for pickup. Customer has 7 days to collect."
    
    elif issue_type == "price_match":
        within_window = ctx.get("within_window", False)
        verified = ctx.get("competitor_verified", False)
        authorized = ctx.get("authorized_retailer", False)
        price_diff = ctx.get("price_difference", 0)
        
        if not within_window:
            return "Not Eligible", "Price match requests must be made within 7 days of purchase."
        
        if not authorized:
            return "Not Eligible", "Price match only available with authorized retailers."
        
        if price_diff < 5:
            return "Not Eligible", "Price difference must be at least $5 to process price match."
        
        if verified and authorized and within_window:
            return "Eligible", "Price match verified. Refund for price difference will be processed (3-5 business days)."
        else:
            return "Escalate", "Price match requires verification of competitor price and item match."
    
    elif issue_type == "subscription_membership":
        if ctx.get("cancellation_requested"):
            return "Eligible", "Subscription cancellation processed. Access continues until end of current billing period. No refund for current period."
        elif ctx.get("unused_portion_refund"):
            return "Escalate", "Unused portion refund requests require supervisor approval and are case-by-case."
        else:
            return "Eligible", "Subscription inquiry handled. Auto-renewal can be cancelled anytime before next billing cycle."
    
    elif issue_type == "digital_product_issue":
        hours = ctx.get("hours_since_purchase", 0)
        accessed = ctx.get("product_accessed", False)
        delivery = ctx.get("delivery_received", False)
        
        if not delivery and hours > 1:
            return "Eligible", "Digital product should be delivered within 15 minutes. Resending delivery email or providing alternative access."
        
        if ctx.get("compatibility_issue") and hours <= 48:
            return "Eligible", "Compatibility issue reported within 48 hours. Refund available if product cannot be used."
        
        if not accessed and hours <= 48:
            return "Eligible", "Refund available within 48 hours if product not accessed, subject to policy terms."
        elif accessed:
            return "Not Eligible", "Digital products are generally non-refundable once accessed. Technical support available for access issues."
        else:
            return "Escalate", "Digital product issue requires technical support review."
    
    elif issue_type == "tax_duty_question":
        if ctx.get("international") and ctx.get("duty_dispute"):
            return "Not Eligible", "Customs duties are set by destination country and cannot be refunded by us. Customer should contact carrier or customs."
        elif ctx.get("tax_exempt_eligible"):
            return "Escalate", "Tax exemption requires valid certificate verification and approval."
        else:
            return "Eligible", "Tax/duty inquiry addressed. Taxes collected per regional requirements. Duties are customer responsibility for international orders."
    
    elif issue_type == "compliance_restriction":
        if ctx.get("restricted_region") or ctx.get("export_control"):
            return "Not Eligible", "Shipping restrictions are legal requirements and cannot be overridden. Alternative products may be available."
        elif ctx.get("age_verified") == False:
            return "Escalate", "Age verification required for age-restricted products. Cannot process order without verification."
        else:
            return "Escalate", "Compliance restriction requires specialist review to determine eligibility and alternatives."
    
    elif issue_type == "size_fit":
        # Similar logic to return_exchange
        days = ctx.get("days_since_delivery", 0)
        condition = ctx.get("condition", "")
        
        if days <= rw and condition in ["unworn", "tried_indoors"]:
            if ctx.get("exchange_requested") and ctx.get("size_available"):
                return "Eligible", f"Size exchange available. Item is within {rw}-day return window and meets condition requirements."
            elif ctx.get("exchange_requested"):
                return "Eligible", f"Size exchange requested. Alternative size may be available or refund can be processed."
            else:
                return "Eligible", f"Return for size/fit issue is eligible within {rw}-day window."
        else:
            return "Not Eligible", f"Size/fit return outside {rw}-day window or condition requirements not met."
    
    elif issue_type == "damaged_item":
        days = ctx.get("days_since_delivery", 0)
        damage_type = ctx.get("damage_type", "")
        
        if damage_type == "defective":
            return "Eligible", "Defective items are always eligible for return or replacement regardless of time."
        elif damage_type == "damaged_in_transit":
            if ctx.get("reported_within_48h"):
                return "Eligible", "Item damaged in transit reported within 48 hours. Immediate replacement or full refund provided."
            else:
                return "Escalate", "Damage in transit reported after 48 hours. Requires review with photographs."
        else:
            return "Escalate", "Damage claim requires review to determine if damage occurred in transit or after delivery."
    
    elif issue_type == "missing_item":
        days = ctx.get("days_since_delivery", 0)
        
        if ctx.get("package_marked_delivered") and not ctx.get("carrier_confirmation"):
            return "Escalate", "Package marked delivered but customer reports not received. Requires investigation with carrier."
        elif ctx.get("partial_order"):
            return "Eligible", "Partial order received. Missing items will be shipped immediately or refunded if unavailable."
        elif days <= 7:
            return "In Progress", "Missing item reported. Investigating with carrier. Resolution within 3-5 business days."
        else:
            return "Eligible", "Missing item confirmed. Replacement shipped or full refund provided."
    
    return "Escalate", "Issue requires manual review."

def generate_customer_response(decision, issue_type, ctx, region, tier):
    """Generate empathetic customer response based on decision"""
    rw = regions[region]["return_window"]
    
    if decision == "Eligible":
        responses = {
            "return_exchange": "I can help you process this return. Since your item meets our return policy requirements, I'll initiate the return process right away.",
            "promotion_pricing": "I'm happy to apply this promotion to your order. The discount will be reflected in your refund or next purchase.",
            "shipping_delay": "I sincerely apologize for the delay. I'm processing a shipping fee refund and expediting your order at no additional cost.",
            "payment_issue": "I've confirmed the duplicate charge and will process your refund immediately. You should see it in 5-10 business days.",
            "warranty_claim": "Your warranty claim has been approved. You can choose between a replacement, repair, or full refund for this manufacturing defect.",
            "loyalty_points": "I've verified your points should have posted and I'm adding them to your account now. You should see them reflected immediately.",
        }
        return responses.get(issue_type, "I can help you with this request. Let me process this for you right away.")
    
    elif decision == "Not Eligible":
        responses = {
            "return_exchange": "I understand you'd like to return this item. Unfortunately, it's outside our return window, but I can explain your options including our warranty policy for defective items.",
            "promotion_pricing": "I wish I could apply that promotion retroactively, but our policy doesn't allow it. However, I can help you with [alternative option] or notify you of future promotions.",
            "shipping_delay": "I understand your concern about the delivery timeline. The current delay is within normal carrier variance, but I'm monitoring it closely and will update you if anything changes.",
            "payment_issue": "I understand your concern. This appears to be an authorization hold rather than an actual charge. These typically release within 5 business days, but I can help verify the status.",
            "warranty_claim": "I understand you're experiencing an issue with your item. Unfortunately, this appears to be normal wear rather than a manufacturing defect, so it's not covered under warranty. However, I can help you explore other options.",
        }
        return responses.get(issue_type, "I understand your request. While I can't accommodate this specific request per our policy, let me explain the policy and see what alternatives we can offer.")
    
    elif decision == "In Progress":
        return "This is currently being processed. I'm monitoring the situation and will keep you updated. You should see resolution within the expected timeframe."
    
    else:  # Escalate
        return "This situation requires additional review to ensure we handle it correctly according to our policies. I'm escalating this to our specialist team who will contact you within 1-2 business days with a resolution."

def build_example():
    """Build a complete training example"""
    region = pick(list(regions.keys()))
    issue = pick(issue_types)
    tier = pick(tiers)
    channel = pick(channels)

    ctx = generate_context(issue, region)
    decision, reason = make_decision(issue, region, ctx, tier)

    # Get relevant policy text (truncated for token efficiency, but include key sections)
    policy_text = get_relevant_policy(issue)
    # Include first 2000 chars of policy (key sections)
    policy_summary = policy_text[:2000] if policy_text else ""

    customer_response = generate_customer_response(decision, issue, ctx, region, tier)

    instruction = (
        "Act as an enterprise e-commerce customer operations assistant. "
        "Follow the provided policy documents strictly and return ONLY valid JSON with keys: "
        "decision, reason, customer_response, next_action, escalation_required. "
        "The customer_response should be written in natural, empathetic language ready to send to the customer."
    )
    
    # Determine next action
    if decision == "Eligible":
        next_action_map = {
            "return_exchange": "Process return and issue return authorization",
            "promotion_pricing": "Apply promotion and process price adjustment",
            "shipping_delay": "Process shipping fee refund and expedite order",
            "payment_issue": "Process refund for duplicate charge",
            "warranty_claim": "Process warranty claim and offer resolution options",
        }
        next_action = next_action_map.get(issue, "Proceed with resolution per policy")
    elif decision == "Not Eligible":
        next_action = "Explain policy and offer alternatives"
    elif decision == "In Progress":
        next_action = "Monitor status and provide updates"
    else:
        next_action = "Escalate to specialist team for review"

    return {
        "instruction": instruction,
        "input": {
            "order_id": oid(),
            "sku": sku(),
            "region": region,
            "channel": channel,
            "customer_tier": tier,
            "issue_type": issue,
            "issue_context": ctx,
            "policy_document": policy_summary,
            "return_window_days": regions[region]["return_window"]
        },
        "output": {
            "decision": decision,
            "reason": reason,
            "customer_response": customer_response,
            "next_action": next_action,
            "escalation_required": decision == "Escalate"
        }
    }

# Generate dataset
print(f"Generating {N} training examples...")
data = [build_example() for _ in range(N)]

# Ensure output directory exists
os.makedirs("data/processed", exist_ok=True)

# Save full dataset
path = f"data/processed/sft_training_dataset_{N}.jsonl"
with open(path, "w", encoding="utf-8") as f:
    for r in data:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Save preview dataset
preview = f"data/processed/sft_training_dataset_preview_100.jsonl"
with open(preview, "w", encoding="utf-8") as f:
    for r in data[:100]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Print statistics
issue_counts = {}
for example in data:
    issue = example["input"]["issue_type"]
    issue_counts[issue] = issue_counts.get(issue, 0) + 1

decision_counts = {}
for example in data:
    decision = example["output"]["decision"]
    decision_counts[decision] = decision_counts.get(decision, 0) + 1

print(f"\nGenerated {len(data)} examples")
print(f"Full dataset: {path}")
print(f"Preview dataset: {preview}")
print(f"\nIssue Type Distribution:")
for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
    print(f"  {issue}: {count}")
print(f"\nDecision Distribution:")
for decision, count in sorted(decision_counts.items(), key=lambda x: -x[1]):
    print(f"  {decision}: {count}")
print(f"\nFirst example:")
print(json.dumps(data[0], indent=2, ensure_ascii=False))
