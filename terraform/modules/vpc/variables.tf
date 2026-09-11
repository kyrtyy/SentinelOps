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
  description = "CIDR blocks for private subnets (EC2 GPU & RDS)"
}

variable "availability_zones" {
  type        = list(string)
  description = "Availability zones to deploy subnets across"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Deployment environment name"
}
