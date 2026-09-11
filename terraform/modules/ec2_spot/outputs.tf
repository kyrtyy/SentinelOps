output "instance_id" {
  value       = aws_instance.gpu_server.id
  description = "EC2 Instance ID"
}

output "private_ip" {
  value       = aws_instance.gpu_server.private_ip
  description = "Private IP of the GPU model server"
}

output "public_ip" {
  value       = aws_instance.gpu_server.public_ip
  description = "Public IP of the GPU model server (if launched in public subnet)"
}

output "security_group_id" {
  value       = aws_security_group.model_server.id
  description = "Security group ID of the model server"
}
