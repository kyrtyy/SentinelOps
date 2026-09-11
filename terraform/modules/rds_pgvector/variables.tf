variable "vpc_id" {
  type        = string
  description = "VPC ID where RDS will be deployed"
}

variable "subnet_ids" {
  type        = list(string)
  description = "Private subnet IDs for DB subnet group"
}

variable "allowed_security_group_ids" {
  type        = list(string)
  description = "Security group IDs allowed to connect to RDS on port 5432"
}

variable "db_name" {
  type        = string
  default     = "sentinelops_runbooks"
  description = "Name of the PostgreSQL database"
}

variable "db_username" {
  type        = string
  default     = "sentinel_admin"
  description = "Database master username"
}

variable "db_password" {
  type        = string
  sensitive   = true
  description = "Database master password"
}

variable "instance_class" {
  type        = string
  default     = "db.t4g.micro"
  description = "RDS instance class (db.t4g.micro is Free Tier eligible)"
}

variable "allocated_storage" {
  type        = number
  default     = 20
  description = "Allocated storage in GB"
}

variable "backup_retention_period" {
  type        = number
  default     = 1
  description = "Backup retention period in days (1 for Free Tier compliance)"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Deployment environment name"
}
