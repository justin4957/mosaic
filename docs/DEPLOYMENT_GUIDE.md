# MOSAIC Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the MOSAIC multi-tiered architecture across development, staging, and production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Development Deployment](#development-deployment)
4. [Staging Deployment](#staging-deployment)
5. [Production Deployment](#production-deployment)
6. [Monitoring & Operations](#monitoring--operations)
7. [Disaster Recovery](#disaster-recovery)
8. [Cost Optimization](#cost-optimization)

## Prerequisites

### Required Tools

```bash
# Install required tools
brew install terraform kubectl helm aws-cli gcloud azure-cli

# Verify installations
terraform version  # >= 1.5.0
kubectl version   # >= 1.27.0
helm version      # >= 3.12.0
aws --version     # >= 2.13.0
gcloud version    # >= 440.0.0
az version        # >= 2.50.0
```

### Cloud Provider Setup

#### AWS Setup
```bash
# Configure AWS CLI
aws configure
AWS Access Key ID: <your-key>
AWS Secret Access Key: <your-secret>
Default region: us-east-1
Default output format: json

# Verify access
aws sts get-caller-identity
```

#### GCP Setup
```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project mosaic-production
gcloud config set compute/region us-central1
```

#### Azure Setup
```bash
# Login to Azure
az login
az account set --subscription "MOSAIC Production"
```

### Repository Setup

```bash
# Clone repository
git clone https://github.com/mosaic/mosaic-platform.git
cd mosaic-platform

# Install dependencies
npm install
go mod download
cargo build --release
```

## Infrastructure Setup

### Terraform Configuration

#### Directory Structure
```
infrastructure/
├── terraform/
│   ├── environments/
│   │   ├── dev/
│   │   ├── staging/
│   │   └── production/
│   ├── modules/
│   │   ├── networking/
│   │   ├── compute/
│   │   ├── storage/
│   │   ├── security/
│   │   └── monitoring/
│   └── global/
```

#### Base Infrastructure Module

```hcl
# File: infrastructure/terraform/modules/base/main.tf

variable "environment" {
  description = "Environment name (dev/staging/production)"
  type        = string
}

variable "region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

# VPC Configuration
module "vpc" {
  source = "../networking"

  environment = var.environment
  cidr_block  = "10.0.0.0/16"

  availability_zones = [
    "${var.region}a",
    "${var.region}b",
    "${var.region}c"
  ]

  public_subnets = [
    "10.0.1.0/24",
    "10.0.2.0/24",
    "10.0.3.0/24"
  ]

  private_subnets = [
    "10.0.10.0/24",
    "10.0.11.0/24",
    "10.0.12.0/24"
  ]

  enable_nat_gateway = true
  enable_vpn_gateway = var.environment == "production"
}

# EKS Cluster
module "eks" {
  source = "../compute/eks"

  environment    = var.environment
  cluster_name   = "mosaic-${var.environment}"
  vpc_id         = module.vpc.vpc_id
  subnet_ids     = module.vpc.private_subnet_ids

  node_groups = {
    general = {
      instance_types = ["t3.medium"]
      min_size       = var.environment == "production" ? 3 : 1
      max_size       = var.environment == "production" ? 10 : 3
      desired_size   = var.environment == "production" ? 3 : 1
    }

    compute = {
      instance_types = ["c5.xlarge"]
      min_size       = var.environment == "production" ? 2 : 0
      max_size       = var.environment == "production" ? 20 : 2
      desired_size   = var.environment == "production" ? 2 : 0

      taints = [{
        key    = "workload"
        value  = "compute"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

# DynamoDB Tables
module "dynamodb" {
  source = "../storage/dynamodb"

  environment = var.environment

  tables = {
    events = {
      hash_key  = "pk"
      range_key = "sk"

      global_secondary_indexes = [{
        name            = "geohash-index"
        hash_key        = "geohash"
        range_key       = "timestamp"
        projection_type = "ALL"
      }]

      stream_enabled   = true
      stream_view_type = "NEW_AND_OLD_IMAGES"

      billing_mode = var.environment == "production" ? "PAY_PER_REQUEST" : "PROVISIONED"

      read_capacity  = var.environment == "production" ? 0 : 5
      write_capacity = var.environment == "production" ? 0 : 5
    }
  }
}

# Redis Cluster
module "redis" {
  source = "../storage/redis"

  environment         = var.environment
  node_type          = var.environment == "production" ? "cache.r6g.xlarge" : "cache.t3.micro"
  number_cache_nodes = var.environment == "production" ? 3 : 1

  subnet_ids         = module.vpc.private_subnet_ids
  security_group_ids = [module.security.redis_sg_id]

  snapshot_retention_limit = var.environment == "production" ? 7 : 1
  automatic_failover       = var.environment == "production"
}
```

### Kubernetes Resources

#### Namespace Configuration

```yaml
# File: k8s/base/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mosaic
  labels:
    name: mosaic
    environment: ${ENVIRONMENT}
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: mosaic-quota
  namespace: mosaic
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
```

#### Deployment Configuration

```yaml
# File: k8s/deployments/coordination-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coordination-service
  namespace: mosaic
spec:
  replicas: ${REPLICAS}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: coordination-service
  template:
    metadata:
      labels:
        app: coordination-service
        version: ${VERSION}
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: coordination-service

      initContainers:
      - name: wait-for-db
        image: busybox:1.35
        command: ['sh', '-c', 'until nc -z dynamodb.amazonaws.com 443; do sleep 1; done']

      containers:
      - name: coordination
        image: mosaic/coordination-service:${VERSION}

        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        - containerPort: 50051
          name: grpc

        env:
        - name: ENVIRONMENT
          value: ${ENVIRONMENT}
        - name: AWS_REGION
          value: ${AWS_REGION}
        - name: REDIS_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: endpoint
        - name: DYNAMODB_TABLE
          value: mosaic-events-${ENVIRONMENT}
        - name: NOISE_RATIO
          value: "${NOISE_RATIO}"
        - name: SESSION_ROTATION_MINUTES
          value: "${SESSION_ROTATION}"

        resources:
          requests:
            cpu: ${CPU_REQUEST}
            memory: ${MEMORY_REQUEST}
          limits:
            cpu: ${CPU_LIMIT}
            memory: ${MEMORY_LIMIT}

        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        volumeMounts:
        - name: config
          mountPath: /etc/mosaic
        - name: secrets
          mountPath: /etc/secrets
          readOnly: true

      volumes:
      - name: config
        configMap:
          name: coordination-config
      - name: secrets
        secret:
          secretName: coordination-secrets
```

## Development Deployment

### Local Development with Docker Compose

```yaml
# File: docker-compose.dev.yml
version: '3.8'

services:
  localstack:
    image: localstack/localstack:latest
    ports:
      - "4566:4566"
    environment:
      - SERVICES=dynamodb,s3,sqs,secretsmanager
      - DEBUG=1
      - DATA_DIR=/tmp/localstack/data
    volumes:
      - "./scripts/localstack:/docker-entrypoint-initaws.d"
      - "localstack_data:/tmp/localstack"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - "redis_data:/data"

  coordination-service:
    build:
      context: ./services/coordination
      dockerfile: Dockerfile.dev
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      - ENVIRONMENT=development
      - AWS_ENDPOINT=http://localstack:4566
      - REDIS_ENDPOINT=redis:6379
      - LOG_LEVEL=debug
    volumes:
      - "./services/coordination:/app"
    depends_on:
      - localstack
      - redis

  mobile-app:
    build:
      context: ./apps/mobile
      dockerfile: Dockerfile.dev
    ports:
      - "19000:19000"  # Expo
      - "19001:19001"
    environment:
      - API_ENDPOINT=http://localhost:8080
      - EXPO_DEVTOOLS_LISTEN_ADDRESS=0.0.0.0
    volumes:
      - "./apps/mobile:/app"
      - "node_modules:/app/node_modules"

volumes:
  localstack_data:
  redis_data:
  node_modules:
```

### Development Scripts

```bash
#!/bin/bash
# File: scripts/dev-deploy.sh

set -e

echo "🚀 Starting MOSAIC Development Environment"

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose required but not installed. Aborting." >&2; exit 1; }

# Set environment variables
export ENVIRONMENT=development
export VERSION=dev-$(git rev-parse --short HEAD)
export NOISE_RATIO=0.6
export SESSION_ROTATION=30

# Start services
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Initialize local DynamoDB tables
aws dynamodb create-table \
  --endpoint-url http://localhost:4566 \
  --table-name mosaic-events-dev \
  --attribute-definitions \
    AttributeName=pk,AttributeType=S \
    AttributeName=sk,AttributeType=S \
  --key-schema \
    AttributeName=pk,KeyType=HASH \
    AttributeName=sk,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  2>/dev/null || true

# Run database migrations
./scripts/run-migrations.sh

# Seed test data
./scripts/seed-data.sh

echo "✅ Development environment ready!"
echo "📱 Mobile app: http://localhost:19000"
echo "🖥️  API: http://localhost:8080"
echo "📊 Metrics: http://localhost:9090"
```

## Staging Deployment

### Staging Environment Setup

```bash
#!/bin/bash
# File: scripts/staging-deploy.sh

set -e

ENVIRONMENT=staging
CLUSTER_NAME=mosaic-staging
REGION=us-east-1

echo "🚀 Deploying to Staging Environment"

# Apply Terraform
cd infrastructure/terraform/environments/staging
terraform init
terraform plan -out=tfplan
terraform apply tfplan

# Get EKS credentials
aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION

# Create namespace
kubectl apply -f k8s/base/namespace.yaml

# Install Helm charts
helm upgrade --install \
  mosaic-staging \
  ./charts/mosaic \
  --namespace mosaic \
  --values ./charts/mosaic/values.staging.yaml \
  --set image.tag=$(git rev-parse --short HEAD) \
  --wait

# Run smoke tests
./scripts/smoke-tests.sh staging

echo "✅ Staging deployment complete!"
```

### Staging Helm Values

```yaml
# File: charts/mosaic/values.staging.yaml
replicaCount: 2

image:
  repository: mosaic
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-staging
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: api-staging.mosaic.privacy
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mosaic-staging-tls
      hosts:
        - api-staging.mosaic.privacy

resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

privacy:
  noiseRatio: 0.6
  sessionRotationMinutes: 45
  defaultTTLHours: 24

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

## Production Deployment

### Multi-Region Production Setup

```hcl
# File: infrastructure/terraform/environments/production/main.tf

terraform {
  required_version = ">= 1.5.0"

  backend "s3" {
    bucket         = "mosaic-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "mosaic-terraform-locks"
    encrypt        = true
  }
}

locals {
  regions = {
    primary   = "us-east-1"
    secondary = "eu-west-1"
    tertiary  = "ap-southeast-1"
  }

  environment = "production"
}

# Primary Region (US East)
module "us_east_1" {
  source = "../../modules/regional"

  providers = {
    aws = aws.us_east_1
  }

  environment = local.environment
  region      = local.regions.primary
  is_primary  = true

  eks_config = {
    cluster_version = "1.27"
    node_groups = {
      general = {
        instance_types = ["c5.2xlarge"]
        min_size       = 5
        max_size       = 50
        desired_size   = 10
      }
      compute = {
        instance_types = ["c5.4xlarge"]
        min_size       = 3
        max_size       = 30
        desired_size   = 5
      }
    }
  }

  dynamodb_global_tables = ["mosaic-events", "mosaic-associations"]
}

# Secondary Region (EU West)
module "eu_west_1" {
  source = "../../modules/regional"

  providers = {
    aws = aws.eu_west_1
  }

  environment = local.environment
  region      = local.regions.secondary
  is_primary  = false

  eks_config = {
    cluster_version = "1.27"
    node_groups = {
      general = {
        instance_types = ["c5.xlarge"]
        min_size       = 3
        max_size       = 20
        desired_size   = 5
      }
    }
  }
}

# Global Accelerator for anycast routing
resource "aws_globalaccelerator_accelerator" "main" {
  name            = "mosaic-production"
  ip_address_type = "IPV4"
  enabled         = true

  attributes {
    flow_logs_enabled = true
    flow_logs_s3_bucket = aws_s3_bucket.logs.bucket
    flow_logs_s3_prefix = "global-accelerator/"
  }
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "api" {
  enabled             = true
  is_ipv6_enabled    = true
  default_root_object = "index.html"

  origin {
    domain_name = aws_globalaccelerator_accelerator.main.dns_name
    origin_id   = "mosaic-api"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2", "TLSv1.3"]
    }
  }

  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD", "OPTIONS"]
    target_origin_id = "mosaic-api"

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "X-Session-Token"]

      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 31536000
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn = aws_acm_certificate.api.arn
    ssl_support_method  = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
}
```

### Production Deployment Pipeline

```yaml
# File: .github/workflows/production-deploy.yml
name: Production Deployment

on:
  push:
    tags:
      - 'v[0-9]+.[0-9]+.[0-9]+'

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: mosaic-production

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}

    steps:
    - uses: actions/checkout@v3

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}

    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1

    - name: Build and push Docker image
      id: build-image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.ref_name }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

  deploy-primary:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: production-primary

    steps:
    - uses: actions/checkout@v3

    - name: Deploy to Primary Region
      run: |
        aws eks update-kubeconfig --name mosaic-production-us-east-1
        helm upgrade --install \
          mosaic-production \
          ./charts/mosaic \
          --namespace mosaic \
          --values ./charts/mosaic/values.production.yaml \
          --set image.tag=${{ github.ref_name }} \
          --set-string image.repository=${{ needs.build-and-push.outputs.image-tag }} \
          --wait \
          --timeout 10m

  deploy-secondary:
    needs: [build-and-push, deploy-primary]
    runs-on: ubuntu-latest
    environment: production-secondary

    strategy:
      matrix:
        region: [eu-west-1, ap-southeast-1]

    steps:
    - uses: actions/checkout@v3

    - name: Deploy to ${{ matrix.region }}
      run: |
        aws eks update-kubeconfig --name mosaic-production-${{ matrix.region }} --region ${{ matrix.region }}
        helm upgrade --install \
          mosaic-production \
          ./charts/mosaic \
          --namespace mosaic \
          --values ./charts/mosaic/values.production-${{ matrix.region }}.yaml \
          --set image.tag=${{ github.ref_name }} \
          --wait \
          --timeout 10m

  verify-deployment:
    needs: [deploy-primary, deploy-secondary]
    runs-on: ubuntu-latest

    steps:
    - name: Run E2E Tests
      run: |
        npm run test:e2e:production

    - name: Check Service Health
      run: |
        ./scripts/health-check.sh production

    - name: Notify Success
      if: success()
      uses: 8398a7/action-slack@v3
      with:
        status: success
        text: 'Production deployment successful: ${{ github.ref_name }}'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## Monitoring & Operations

### Prometheus Configuration

```yaml
# File: k8s/monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    scrape_configs:
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

    - job_name: 'mosaic-services'
      static_configs:
      - targets:
        - coordination-service:9090
        - proximity-service:9090
        - association-service:9090

    alerting:
      alertmanagers:
      - static_configs:
        - targets:
          - alertmanager:9093

    rule_files:
      - '/etc/prometheus/rules/*.yml'
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "MOSAIC Production Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Privacy Metrics",
        "targets": [
          {
            "expr": "mosaic_noise_ratio",
            "legendFormat": "Noise Ratio"
          },
          {
            "expr": "mosaic_obfuscated_events_total",
            "legendFormat": "Obfuscated Events"
          }
        ]
      },
      {
        "title": "Association Metrics",
        "targets": [
          {
            "expr": "rate(mosaic_associations_created[5m])",
            "legendFormat": "Associations/sec"
          }
        ]
      },
      {
        "title": "System Health",
        "targets": [
          {
            "expr": "up{job=\"mosaic-services\"}",
            "legendFormat": "{{instance}}"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# File: k8s/monitoring/alert-rules.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: monitoring
data:
  mosaic-alerts.yml: |
    groups:
    - name: mosaic-critical
      interval: 30s
      rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
          description: "Error rate is {{ $value }} (threshold: 0.05)"

      - alert: LowNoiseRatio
        expr: mosaic_noise_ratio < 0.3
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: Noise ratio below minimum threshold
          description: "Noise ratio is {{ $value }} (minimum: 0.3)"

      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: Container memory usage high
          description: "Memory usage is {{ $value }}% of limit"

      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: Pod crash looping
          description: "Pod {{ $labels.pod }} is crash looping"
```

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# File: scripts/backup.sh

set -e

ENVIRONMENT=${1:-production}
BACKUP_BUCKET="mosaic-backups-${ENVIRONMENT}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🔄 Starting backup for ${ENVIRONMENT}"

# Backup DynamoDB tables
for TABLE in mosaic-events mosaic-associations; do
  echo "Backing up DynamoDB table: ${TABLE}"
  aws dynamodb create-backup \
    --table-name ${TABLE}-${ENVIRONMENT} \
    --backup-name ${TABLE}-${TIMESTAMP}
done

# Export Redis snapshot
echo "Creating Redis snapshot"
aws elasticache create-snapshot \
  --cache-cluster-id mosaic-redis-${ENVIRONMENT} \
  --snapshot-name redis-${TIMESTAMP}

# Backup Kubernetes resources
echo "Backing up Kubernetes resources"
kubectl get all -n mosaic -o yaml > k8s-resources-${TIMESTAMP}.yaml
aws s3 cp k8s-resources-${TIMESTAMP}.yaml s3://${BACKUP_BUCKET}/k8s/

# Backup secrets (encrypted)
echo "Backing up secrets"
kubectl get secrets -n mosaic -o yaml | \
  gpg --encrypt --recipient backup@mosaic.privacy > secrets-${TIMESTAMP}.yaml.gpg
aws s3 cp secrets-${TIMESTAMP}.yaml.gpg s3://${BACKUP_BUCKET}/secrets/

echo "✅ Backup complete: ${TIMESTAMP}"
```

### Recovery Procedures

```bash
#!/bin/bash
# File: scripts/disaster-recovery.sh

set -e

BACKUP_TIMESTAMP=$1
ENVIRONMENT=${2:-production}

if [ -z "$BACKUP_TIMESTAMP" ]; then
  echo "Usage: $0 <backup-timestamp> [environment]"
  exit 1
fi

echo "🚨 Starting disaster recovery from backup: ${BACKUP_TIMESTAMP}"

# Restore DynamoDB tables
for TABLE in mosaic-events mosaic-associations; do
  echo "Restoring DynamoDB table: ${TABLE}"

  # Find backup ARN
  BACKUP_ARN=$(aws dynamodb list-backups \
    --table-name ${TABLE}-${ENVIRONMENT} \
    --backup-type USER \
    --query "BackupSummaries[?BackupName=='${TABLE}-${BACKUP_TIMESTAMP}'].BackupArn" \
    --output text)

  if [ -n "$BACKUP_ARN" ]; then
    aws dynamodb restore-table-from-backup \
      --backup-arn ${BACKUP_ARN} \
      --target-table-name ${TABLE}-${ENVIRONMENT}-restored
  fi
done

# Restore Redis from snapshot
echo "Restoring Redis cluster"
aws elasticache create-cache-cluster \
  --cache-cluster-id mosaic-redis-${ENVIRONMENT}-restored \
  --snapshot-name redis-${BACKUP_TIMESTAMP} \
  --cache-node-type cache.r6g.xlarge \
  --engine redis \
  --num-cache-nodes 3

# Restore Kubernetes resources
echo "Restoring Kubernetes resources"
aws s3 cp s3://mosaic-backups-${ENVIRONMENT}/k8s/k8s-resources-${BACKUP_TIMESTAMP}.yaml .
kubectl apply -f k8s-resources-${BACKUP_TIMESTAMP}.yaml

echo "✅ Disaster recovery complete"
```

### Failover Procedures

```bash
#!/bin/bash
# File: scripts/failover.sh

set -e

PRIMARY_REGION="us-east-1"
SECONDARY_REGION="eu-west-1"

echo "🔄 Initiating failover from ${PRIMARY_REGION} to ${SECONDARY_REGION}"

# Update Route53 weighted routing
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456789 \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.mosaic.privacy",
        "Type": "A",
        "SetIdentifier": "Primary",
        "Weight": 0,
        "AliasTarget": {
          "HostedZoneId": "Z215JYRZR8TBV5",
          "DNSName": "us-east-1.elb.amazonaws.com",
          "EvaluateTargetHealth": false
        }
      }
    },
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.mosaic.privacy",
        "Type": "A",
        "SetIdentifier": "Secondary",
        "Weight": 100,
        "AliasTarget": {
          "HostedZoneId": "Z3F0SRJ5LGBH90",
          "DNSName": "eu-west-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# Scale up secondary region
aws eks update-nodegroup-config \
  --cluster-name mosaic-production-${SECONDARY_REGION} \
  --nodegroup-name general \
  --scaling-config desiredSize=20,minSize=10,maxSize=50 \
  --region ${SECONDARY_REGION}

# Notify team
./scripts/notify-slack.sh "Failover initiated: Primary (${PRIMARY_REGION}) → Secondary (${SECONDARY_REGION})"

echo "✅ Failover complete"
```

## Cost Optimization

### Resource Optimization

```yaml
# File: k8s/optimization/vertical-pod-autoscaler.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: coordination-service-vpa
  namespace: mosaic
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: coordination-service
  updatePolicy:
    updateMode: Auto
  resourcePolicy:
    containerPolicies:
    - containerName: coordination
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: 2000m
        memory: 4Gi
      controlledResources: ["cpu", "memory"]
```

### Cost Monitoring

```python
#!/usr/bin/env python3
# File: scripts/cost-monitor.py

import boto3
import json
from datetime import datetime, timedelta

def get_daily_costs(days=7):
    ce = boto3.client('ce')

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start_date.strftime('%Y-%m-%d'),
            'End': end_date.strftime('%Y-%m-%d')
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        GroupBy=[
            {'Type': 'DIMENSION', 'Key': 'SERVICE'},
            {'Type': 'TAG', 'Key': 'Environment'}
        ]
    )

    return response['ResultsByTime']

def analyze_costs(cost_data):
    total_cost = 0
    service_costs = {}

    for day in cost_data:
        for group in day['Groups']:
            service = group['Keys'][0]
            cost = float(group['Metrics']['UnblendedCost']['Amount'])

            if service not in service_costs:
                service_costs[service] = 0
            service_costs[service] += cost
            total_cost += cost

    # Sort by cost
    sorted_services = sorted(
        service_costs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"Total cost (last 7 days): ${total_cost:.2f}")
    print("\nTop services by cost:")
    for service, cost in sorted_services[:10]:
        percentage = (cost / total_cost) * 100
        print(f"  {service}: ${cost:.2f} ({percentage:.1f}%)")

    # Alert if costs exceed threshold
    daily_average = total_cost / 7
    if daily_average > 5000:  # $5000/day threshold
        send_alert(f"High daily cost detected: ${daily_average:.2f}")

def send_alert(message):
    sns = boto3.client('sns')
    sns.publish(
        TopicArn='arn:aws:sns:us-east-1:123456789012:cost-alerts',
        Subject='MOSAIC Cost Alert',
        Message=message
    )

if __name__ == '__main__':
    costs = get_daily_costs()
    analyze_costs(costs)
```

### Spot Instance Configuration

```yaml
# File: k8s/optimization/spot-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
  namespace: kube-system
data:
  nodes.max-node-provision-time: "15m"
  scale-down-delay-after-add: "10m"
  scale-down-unneeded-time: "10m"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  template:
    spec:
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.27.0
        name: cluster-autoscaler
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/mosaic-production
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
        - --aws-use-static-instance-list=false
        - --max-node-provision-time=15m
        env:
        - name: AWS_REGION
          value: us-east-1
```

## Security Hardening

### Network Policies

```yaml
# File: k8s/security/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mosaic-network-policy
  namespace: mosaic
spec:
  podSelector:
    matchLabels:
      app: coordination-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: mosaic
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 50051
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: mosaic
    ports:
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 443   # HTTPS
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
```

### Pod Security Policies

```yaml
# File: k8s/security/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: mosaic-restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  supplementalGroups:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

## Troubleshooting

### Common Issues and Solutions

```bash
#!/bin/bash
# File: scripts/troubleshoot.sh

echo "🔍 MOSAIC Troubleshooting Script"

# Check pod status
echo -e "\n📦 Pod Status:"
kubectl get pods -n mosaic -o wide

# Check recent events
echo -e "\n📅 Recent Events:"
kubectl get events -n mosaic --sort-by='.lastTimestamp' | head -20

# Check service endpoints
echo -e "\n🔗 Service Endpoints:"
kubectl get endpoints -n mosaic

# Check resource usage
echo -e "\n💻 Resource Usage:"
kubectl top nodes
kubectl top pods -n mosaic

# Check logs for errors
echo -e "\n📝 Recent Errors:"
kubectl logs -n mosaic -l app=coordination-service --tail=50 | grep -i error

# Check network connectivity
echo -e "\n🌐 Network Connectivity:"
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -n mosaic -- bash -c "
  echo 'Testing DNS resolution...'
  nslookup dynamodb.amazonaws.com
  echo 'Testing Redis connectivity...'
  nc -zv redis-cluster.mosaic.svc.cluster.local 6379
  echo 'Testing external connectivity...'
  curl -I https://api.mosaic.privacy
"

# Database connectivity
echo -e "\n🗄️ Database Status:"
aws dynamodb describe-table --table-name mosaic-events-production --query 'Table.TableStatus'
aws elasticache describe-cache-clusters --cache-cluster-id mosaic-redis-production --query 'CacheClusters[0].CacheClusterStatus'
```

---

This deployment guide provides comprehensive instructions for deploying and managing the MOSAIC platform across all environments, from local development to global production scale.