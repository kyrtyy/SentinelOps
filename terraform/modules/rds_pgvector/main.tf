# RDS PostgreSQL Module with pgvector for SentinelOps Runbooks Knowledge Base

resource "aws_db_subnet_group" "rds" {
  name        = "sentinelops-rds-subnet-group-${var.environment}"
  subnet_ids  = var.subnet_ids
  description = "Subnet group for SentinelOps RDS pgvector instance"

  tags = {
    Name        = "sentinelops-rds-subnet-group-${var.environment}"
    Environment = var.environment
  }
}

resource "aws_db_parameter_group" "pgvector" {
  name        = "sentinelops-pgvector-params-${var.environment}"
  family      = "postgres16"
  description = "Custom parameter group enabling pgvector extension"

  parameter {
    name         = "shared_preload_libraries"
    value        = "pgvector"
    apply_method = "pending-reboot"
  }

  tags = {
    Name        = "sentinelops-pgvector-params-${var.environment}"
    Environment = var.environment
  }
}

resource "aws_security_group" "rds" {
  name        = "sentinelops-rds-sg-${var.environment}"
  description = "Controls database access to pgvector runbook store"
  vpc_id      = var.vpc_id

  ingress {
    description     = "PostgreSQL access from model server & lambda"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = var.allowed_security_group_ids
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "sentinelops-rds-sg-${var.environment}"
    Environment = var.environment
  }
}

resource "aws_db_instance" "postgres" {
  identifier             = "sentinelops-runbooks-${var.environment}"
  engine                 = "postgres"
  engine_version         = "16.3"
  instance_class         = var.instance_class
  allocated_storage      = var.allocated_storage
  max_allocated_storage  = 100
  storage_type           = "gp3"

  db_name  = var.db_name
  username = var.db_username
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.rds.name
  parameter_group_name   = aws_db_parameter_group.pgvector.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  skip_final_snapshot    = true
  deletion_protection    = false
  publicly_accessible    = false

  backup_retention_period = 7

  tags = {
    Name        = "sentinelops-runbooks-pgvector-${var.environment}"
    Environment = var.environment
  }
}
