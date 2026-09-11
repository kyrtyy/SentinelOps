variable "vpc_id" {
  type        = string
  description = "VPC ID where the Lambda will be attached"
}

variable "subnet_ids" {
  type        = list(string)
  description = "Private subnet IDs for the Lambda VPC interface"
}

variable "model_server_ip" {
  type        = string
  description = "Private IP address of the SentinelOps EC2 GPU model server"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Deployment environment name"
}
