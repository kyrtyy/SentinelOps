terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "SentinelOps"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# 1. Networking (VPC, Public & Private Subnets)
module "vpc" {
  source = "./modules/vpc"

  vpc_cidr             = var.vpc_cidr
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  availability_zones   = var.availability_zones
  environment          = var.environment
}

# 2. EC2 Spot GPU Instance (NVIDIA A10G 24GB on g5.xlarge)
module "ec2_spot" {
  source = "./modules/ec2_spot"

  vpc_id            = module.vpc.vpc_id
  vpc_cidr          = var.vpc_cidr
  subnet_id         = module.vpc.private_subnet_ids[0]
  instance_type     = var.ec2_instance_type
  key_name          = var.ec2_key_name
  admin_cidr_blocks = var.admin_cidr_blocks
  environment       = var.environment
}

# 3. RDS PostgreSQL with pgvector for Runbook Knowledge Base
module "rds_pgvector" {
  source = "./modules/rds_pgvector"

  vpc_id                     = module.vpc.vpc_id
  subnet_ids                 = module.vpc.private_subnet_ids
  allowed_security_group_ids = [module.ec2_spot.security_group_id]
  db_name                    = var.db_name
  db_username                = var.db_username
  db_password                = var.db_password
  instance_class             = var.db_instance_class
  environment                = var.environment
}

# 4. EventBridge + AWS Lambda for CloudWatch Alarm Ingestion
module "eventbridge_lambda" {
  source = "./modules/eventbridge_lambda"

  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnet_ids
  model_server_ip = module.ec2_spot.private_ip
  environment     = var.environment
}
