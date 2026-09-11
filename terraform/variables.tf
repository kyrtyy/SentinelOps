variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS deployment region"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Deployment environment name (e.g. production, staging, dev)"
}

variable "vpc_cidr" {
  type        = string
  default     = "10.0.0.0/16"
  description = "CIDR block for the SentinelOps VPC"
}

variable "public_subnet_cidrs" {
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
  description = "CIDR blocks for public subnets"
}

variable "private_subnet_cidrs" {
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.20.0/24"]
  description = "CIDR blocks for private subnets"
}

variable "availability_zones" {
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
  description = "Availability zones to deploy into"
}

variable "ec2_instance_type" {
  type        = string
  default     = "g5.xlarge"
  description = "EC2 GPU instance type (g5.xlarge provides 1x NVIDIA A10G 24GB)"
}

variable "ec2_key_name" {
  type        = string
  default     = ""
  description = "Optional EC2 Key Pair for SSH"
}

variable "admin_cidr_blocks" {
  type        = list(string)
  default     = ["0.0.0.0/0"]
  description = "CIDR blocks allowed for administrative SSH access"
}

variable "db_name" {
  type        = string
  default     = "sentinelops_runbooks"
  description = "RDS database name"
}

variable "db_username" {
  type        = string
  default     = "sentinel_admin"
  description = "RDS master username"
}

variable "db_password" {
  type        = string
  sensitive   = true
  description = "RDS master password"
}

variable "db_instance_class" {
  type        = string
  default     = "db.t4g.medium"
  description = "RDS instance class"
}
