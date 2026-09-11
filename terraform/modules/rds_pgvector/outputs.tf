output "endpoint" {
  value       = aws_db_instance.postgres.endpoint
  description = "Connection endpoint for the RDS PostgreSQL database"
}

output "address" {
  value       = aws_db_instance.postgres.address
  description = "Database hostname"
}

output "db_name" {
  value       = aws_db_instance.postgres.db_name
  description = "Database name"
}

output "security_group_id" {
  value       = aws_security_group.rds.id
  description = "Security group ID of the RDS instance"
}
