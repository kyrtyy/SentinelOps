output "lambda_arn" {
  value       = aws_lambda_function.triage_trigger.arn
  description = "ARN of the SentinelOps triage trigger Lambda"
}

output "lambda_function_name" {
  value       = aws_lambda_function.triage_trigger.function_name
  description = "Name of the triage trigger Lambda function"
}

output "eventbridge_rule_arn" {
  value       = aws_cloudwatch_event_rule.incident_rule.arn
  description = "ARN of the EventBridge alarm capture rule"
}
