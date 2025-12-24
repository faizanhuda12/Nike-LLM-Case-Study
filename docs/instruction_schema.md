# Instruction Fine-Tuning Schema

Each training sample follows a strict structure:

## instruction
Natural language task definition the model must follow.

## input
Structured context including:
- customer_region
- order_status
- days_since_delivery
- product_category
- issue_description
- applicable_policy_rules

## output
Structured JSON with:
- decision (eligible / not_eligible / escalate)
- reasoning (concise, policy-grounded)
- customer_response (brand-aligned)
- next_action (enum)
