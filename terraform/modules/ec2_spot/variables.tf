variable "vpc_id" {
  type        = string
  description = "VPC ID where the EC2 instance will be launched"
}

variable "vpc_cidr" {
  type        = string
  description = "VPC CIDR block allowed to call the model server API"
}

variable "subnet_id" {
  type        = string
  description = "Subnet ID for the GPU instance"
}

variable "instance_type" {
  type        = string
  default     = "g5.xlarge"
  description = "EC2 GPU instance type (g5.xlarge contains NVIDIA A10G 24GB)"
}

variable "spot_max_price" {
  type        = string
  default     = null
  description = "Maximum spot price per hour. Defaults to on-demand price if null."
}

variable "ami_id" {
  type        = string
  default     = ""
  description = "Custom AMI ID. If empty, automatically selects latest AWS Deep Learning AMI."
}

variable "key_name" {
  type        = string
  default     = ""
  description = "EC2 Key Pair name for SSH"
}

variable "admin_cidr_blocks" {
  type        = list(string)
  default     = ["0.0.0.0/0"]
  description = "CIDR blocks allowed for administrative SSH access"
}

variable "root_volume_size" {
  type        = number
  default     = 100
  description = "Size of EBS root volume in GB"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Deployment environment name"
}
