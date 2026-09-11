# EC2 Spot GPU Instance Module for SentinelOps Model Serving

resource "aws_iam_role" "ec2_role" {
  name = "sentinelops-ec2-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "sentinelops-ec2-role-${var.environment}"
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "profile" {
  name = "sentinelops-ec2-profile-${var.environment}"
  role = aws_iam_role.ec2_role.name
}

resource "aws_security_group" "model_server" {
  name        = "sentinelops-model-server-sg-${var.environment}"
  description = "Security group for SentinelOps GPU model server"
  vpc_id      = var.vpc_id

  ingress {
    description = "API access on port 8000"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description = "SSH administrative access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.admin_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "sentinelops-model-server-sg-${var.environment}"
    Environment = var.environment
  }
}

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04)*"]
  }
}

resource "aws_instance" "gpu_server" {
  ami                  = var.ami_id != "" ? var.ami_id : data.aws_ami.deep_learning.id
  instance_type        = var.instance_type
  subnet_id            = var.subnet_id
  vpc_security_group_ids = [aws_security_group.model_server.id]
  iam_instance_profile = aws_iam_instance_profile.profile.name
  key_name             = var.key_name != "" ? var.key_name : null

  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = var.spot_max_price
    }
  }

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = file("${path.module}/user_data.sh")

  tags = {
    Name        = "sentinelops-gpu-server-${var.environment}"
    Environment = var.environment
  }
}
