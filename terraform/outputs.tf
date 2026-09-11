output "vpc_id" {
  value       = module.vpc.vpc_id
  description = "SentinelOps VPC ID"
}

output "gpu_server_private_ip" {
  value       = module.ec2_spot.private_ip
  description = "Private IP of the EC2 GPU model server"
}

output "gpu_server_public_ip" {
  value       = module.ec2_spot.public_ip
  description = "Public IP of the EC2 GPU model server (if applicable)"
}

output "rds_endpoint" {
  value       = module.rds_pgvector.endpoint
  description = "RDS PostgreSQL connection endpoint"
}

output "lambda_trigger_arn" {
  value       = module.eventbridge_lambda.lambda_arn
  description = "ARN of the triage trigger Lambda function"
}

output "eventbridge_rule_arn" {
  value       = module.eventbridge_lambda.eventbridge_rule_arn
  description = "ARN of the EventBridge rule capturing incident alarms"
}
