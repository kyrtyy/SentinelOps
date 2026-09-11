#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== SentinelOps AWS Infrastructure Deployment ==="

if [ ! -f "terraform.tfvars" ]; then
    echo "Warning: terraform.tfvars not found!"
    echo "Creating terraform.tfvars from terraform.tfvars.example..."
    cp terraform.tfvars.example terraform.tfvars
    echo "Please edit terraform.tfvars with your custom database password and AWS settings."
    exit 1
fi

echo "1. Initializing Terraform..."
terraform init

echo "2. Validating configuration..."
terraform validate

echo "3. Planning deployment..."
terraform plan -out=tfplan

echo ""
read -p "Do you want to apply this deployment to AWS? (y/N): " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "4. Applying Terraform plan..."
    terraform apply tfplan
    echo "Deployment complete!"
else
    echo "Deployment cancelled."
fi
